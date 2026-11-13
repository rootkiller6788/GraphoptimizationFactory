"""对抗用例：**关键不变量**的运行时核对。

## 与 `acceptance.py` 的分工

- `acceptance.py` 的**门槛**是**可调**的：阈值写在 `config.acceptance` 里，
  可以被调松（那正是「门槛」的意义 —— 它是达标线）。
- 这里的**对抗用例**是**不可调**的：要么成立要么不成立，配置里没有它的阈值。
  它们不是达标线，是「**证明这个机制真的在工作**」。
  ⇒ 一条对抗用例红了，意味着**代码或产物自相矛盾**，不是「这次没达标」。

## ⚠️ 一条口径纪律（本项目踩过）

`adversarial.json` 里那 7 个**非法变体**（`relu_to_identity` / `drop_bias` /
`bias_epsilon(ε)`×5）是**被测量的实验对象** —— 它们是**数据**，不是「对抗用例」。

第一版把它们称作「对抗用例」，于是报告里写「7 条对抗用例」，
而 `L5` 各项目的对抗用例是 **24~38 条**（都是不变量断言）——
**同一个词指了两个不同的东西，读者没法从报告内部发现。**
⇒ 本文件用「对抗用例」只指不变量断言；那 7 个一律叫**非法变体**。

## 三条写用例的纪律

1. **用例红了，第一反应是「用例是不是写错了」。** 一个自己写错的用例比没有用例更糟 ——
   它会让人去修一个不存在的问题。
2. **用例自己抛异常也必须算失败**，不许吞（吞掉就成了一条静默失效的通道）。
3. **用例要能红。** 每条用例的 `claim` 都要写清「它在什么情况下会红」，
   答不上来的那条删掉 —— 一条恒真的用例与没有用例在输出里长得一模一样。
"""

from __future__ import annotations

import math
from typing import Any, Callable

from .ladder import mean_gain
from .pareto import dominates, hypervolume, non_dominated_indices
from .search import permutation_counts

#: 用例名 → 求值函数。返回 `(ok, detail, observed)`。
_CASES: dict[str, Callable[[dict[str, Any]], tuple[bool, str, Any]]] = {}


def _case(name: str):
    def deco(fn):
        _CASES[name] = fn
        return fn
    return deco


# --------------------------------------------------------------------- 小工具
def _cfg(p: dict[str, Any]) -> dict[str, Any]:
    return p["config.json"]["resolved"]


def _rules(p: dict[str, Any]) -> list[dict[str, Any]]:
    return p["lineage.json"]["rules"]


def _legal(p: dict[str, Any]) -> list[dict[str, Any]]:
    return [r for r in _rules(p) if r["kind"] == "legal"]


def _illegal(p: dict[str, Any]) -> list[dict[str, Any]]:
    return [r for r in _rules(p) if r["kind"] == "illegal"]


def _illegal_entries(p: dict[str, Any]) -> list[tuple[str, dict[str, bool], float]]:
    """**全部**非法重写 → `(名字, accepted_at, drift)`。

    ⚠️ 非法重写有**两个来源**，必须并起来：

    - `lineage.json.rules` 里 2 条对照规则（`relu_to_identity` / `drop_bias`），
      键名是 `rule`；
    - `adversarial.json.variants` 里 5 档 `bias_epsilon(ε)`，键名是 `variant`。

    第一版只取了前者 ⇒ 「窗口内全部被拒」这条断言只覆盖了 **2/7** 个非法重写，
    而它印在报告里看起来像「7 个都验过了」。
    **同一个 bug 在 `safe_window_matches_the_raw_drifts` 里修过一次，这里漏修了 ——
    两处都从「原始漂移/接受矩阵」取数，本该共用同一个集合。**
    """
    out: list[tuple[str, dict[str, bool], float]] = [
        (r["rule"], r["accepted_at"], float(r["drift"])) for r in _illegal(p)]
    out += [(v["variant"], v["accepted_at"], float(v["drift"]))
            for v in p["adversarial.json"]["variants"]]
    return out


def _tolerances(p: dict[str, Any]) -> list[float]:
    return [float(x) for x in _cfg(p)["ladder"]["tolerances"]]


def _rungs(p: dict[str, Any]) -> list[dict[str, Any]]:
    return p["ladder.json"]["rungs"]


def _rung(p: dict[str, Any], strategy: str, tol: float) -> dict[str, Any]:
    for r in _rungs(p):
        if r["strategy"] == strategy and r["tolerance"] == tol:
            return r
    raise KeyError(f"没有这一级：{strategy}@{tol}")


def _tau0(p: dict[str, Any]) -> float:
    return min(_tolerances(p))


def _at(accepted_at: dict[str, bool], tol: float) -> bool:
    """容差 → `accepted_at` 的键，判「这一档接受了吗」。

    ⚠️ 键是 `str(t)` 生成的（见 `adversarial.py`），**不是** `f"{t:g}"`：
    对 `0.0` 前者给 `'0.0'`、后者给 `'0'` —— 用 `:g` 查 τ=0 那一级会**永远查不到**。
    于是「τ=0 时非法重写全部被拒」这句话里，τ=0 那一格**根本没被检查**。
    **一个永远查不到的键，与「查到了且通过」在输出里长得一模一样。**
    """
    return bool(accepted_at.get(str(float(tol))))


def _all_plans(p: dict[str, Any]) -> list[tuple[str, ...]]:
    """穷举空间：`P(n,1)+P(n,2)+…+P(n,max_depth)`，**有序不重复**。"""
    names = list(_cfg(p)["search"]["rule_order"])
    depth = int(_cfg(p)["search"]["max_depth"])
    out: list[tuple[str, ...]] = []

    def rec(prefix: tuple[str, ...]) -> None:
        if prefix:
            out.append(prefix)
        if len(prefix) == depth:
            return
        for n in names:
            if n not in prefix:
                rec(prefix + (n,))

    rec(())
    return out


def _trial_objectives(cell: dict[str, Any]) -> list[list[float]]:
    return [list(t["objectives"]) for t in cell.get("frontier", [])]


def _cells(p: dict[str, Any]):
    for r in _rungs(p):
        for name in sorted(r["cells"]):
            yield r, r["cells"][name]


# ===================================================================== 空间与身份
@_case("space_size_matches_the_permutation_count")
def _i_space(p):
    """空间大小必须等于排列数之和。

    红的情况：改了规则数或最大深度，却没同步空间枚举 ——
    那样「oracle 枚举全空间」这句话就是假的，而它看起来仍然像 oracle。
    """
    n = len(_cfg(p)["search"]["rule_order"])
    d = int(_cfg(p)["search"]["max_depth"])
    counts = permutation_counts(n, d)
    total = sum(counts)
    size = int(p["space.json"]["space_size"])
    enum = len(_all_plans(p))
    ok = total == size == enum
    return (ok,
            f"P({n},1..{d}) = {' + '.join(str(c) for c in counts)} = {total}；"
            f"space.json 报 {size}；独立枚举得 {enum}",
            {"permutation_counts": counts, "declared": size, "enumerated": enum})


@_case("plan_ids_are_injective")
def _i_planid(p):
    """`plan_id` 必须与计划一一对应（截断到 10 位十六进制也不能碰撞）。

    红的情况：`Plan.id` 的摘要口径变了（例如只取规则集合、丢掉顺序）——
    那样 `[fusion, cse]` 与 `[cse, fusion]` 会得到同一个 id，
    去重会把两个不同的计划当成一个，而前沿看起来更干净。
    """
    plans = _all_plans(p)
    ids = ["p:" + __import__("hashlib").sha256(
        "|".join(r).encode("utf-8")).hexdigest()[:10] for r in plans]
    ok = len(set(ids)) == len(ids) == len(plans)
    return (ok,
            f"{len(plans)} 个计划产生 {len(set(ids))} 个不同的 plan_id"
            f"（要求一一对应）",
            {"n_plans": len(plans), "n_ids": len(set(ids))})


@_case("every_rule_fires_at_least_once")
def _i_fires(p):
    """8 条合法规则**每条都要至少命中一次**。

    红的情况：形状预演用了错的规则（本项目真踩过：`matmul_reassociate`
    拿广播规则去预演矩阵乘，于是**静默命中 0 次**）。
    「命中 0 次」不会报错，只会让这条规则在报告里看起来「没用」。
    """
    dead = [r["rule"] for r in _legal(p) if int(r["hits"]) <= 0]
    hits = {r["rule"]: int(r["hits"]) for r in _legal(p)}
    return (not dead,
            f"8 条合法规则的命中数 {hits}；命中 0 次的有 {dead}",
            hits)


@_case("no_rule_is_a_noop")
def _i_noop(p):
    """规则被调用、但图**逐位没变**的次数必须是 0。

    红的情况：一条规则写成了「返回输入的副本」—— 它会计入命中、
    让「这条规则有用」看起来成立，而实际上什么都没做。
    """
    d = p["space.json"]["evaluator"]
    n = int(d["n_noop"])
    return (n == 0,
            f"noop（调用后图逐位不变）次数 = {n}",
            {"n_noop": n, "n_bit_identical": d["n_bit_identical"]})


@_case("rule_drift_and_bit_identity_agree")
def _i_identity(p):
    """合法规则：`bit_identical` 必须与 `drift == 0` **互为充要**。

    红的情况：漂移用容差判定（`drift <= τ` 就算 0）——
    那样「逐位等价」这句话会随容差变化，而它本该是一个与配置无关的事实。
    """
    bad = []
    for r in _legal(p):
        if bool(r["bit_identical"]) != (float(r["drift"]) == 0.0):
            bad.append(r["rule"])
    illegal_bit = [r["rule"] for r in _illegal(p) if r["bit_identical"]]
    ok = not bad and not illegal_bit
    return (ok,
            f"合法规则里 bit_identical 与 drift==0 不一致的：{bad}；"
            f"非法规则里被标成逐位等价的：{illegal_bit}",
            {"inconsistent": bad, "illegal_bit_identical": illegal_bit})


@_case("rule_set_is_partitioned_into_legal_and_control")
def _i_partition(p):
    """规则表 = 8 条合法 + 2 条非法对照，且与配置里的规则名一致。

    红的情况：加了一条规则却忘了在 `rule_table` 里给它分类 ——
    它会**默默消失**，而「8 条合法重写」这句话仍然印在报告里。
    """
    cfg_names = set(_cfg(p)["search"]["rule_order"])
    legal = {r["rule"] for r in _legal(p)}
    ctrl = {r["rule"] for r in _illegal(p)}
    declared = set(p["lineage.json"]["legal_names"])
    ok = (legal == declared == cfg_names) and len(legal) == 8 and len(ctrl) == 2
    return (ok,
            f"合法 {len(legal)} 条 {sorted(legal)}；非法对照 {len(ctrl)} 条 {sorted(ctrl)}；"
            f"配置声明的规则名 {sorted(cfg_names)}",
            {"legal": sorted(legal), "control": sorted(ctrl),
             "declared": sorted(declared), "configured": sorted(cfg_names)})


# ===================================================================== 容差崖
@_case("safe_window_matches_the_raw_drifts")
def _i_window_raw(p):
    """安全窗口的端点必须**就是**原始漂移的极值，不是另算的一对数。

    ⚠️ 非法漂移有**两个来源**，必须并起来：`lineage.rules` 里只有 2 条对照规则
    （`relu_to_identity` / `drop_bias`），`bias_epsilon(ε)` 那 5 档在
    `adversarial.variants` 里。第一版只取了前者，于是「非法最小漂移」
    被算成 `drop_bias` 的 0.805 —— 与窗口上界 1.519e-12 差 12 个数量级。
    **这不是数据错了，是用例只看了半个集合。**
    """
    w = p["expectations.json"]["derived"]["safe_window"]
    lo = max(float(r["drift"]) for r in _legal(p))
    illegal = [drift for _, _, drift in _illegal_entries(p)]
    hi = min(illegal)
    ok = (abs(lo - float(w["lo"])) < 1e-30 and abs(hi - float(w["hi"])) < 1e-30
          and bool(w["nonempty"]) == (lo < hi))
    return (ok,
            f"合法最大漂移 {lo:.6e} vs 窗口下界 {w['lo']:.6e}；"
            f"非法最小漂移 {hi:.6e}（{len(illegal)} 个非法漂移取最小）"
            f" vs 窗口上界 {w['hi']:.6e}",
            {"lo_raw": lo, "hi_raw": hi, "window": [w["lo"], w["hi"]],
             "n_illegal_drifts": len(illegal)})


@_case("illegal_rewrites_are_all_rejected_below_the_window")
def _i_illegal_below(p):
    """**窗口上界以下**的每一级容差，都必须拒掉**全部**非法重写。

    红的情况：容差崖算错了（例如漂移用了归一化后的值）——
    那样某个非法变体会在窗口内被放过去，而它看起来像「一条可接受的重写」。

    ⚠️ 「全部非法重写」必须包含 `adversarial.json` 里那 5 档 `bias_epsilon(ε)`
    （见 `_illegal_entries`）。只查 `lineage.json` 的 2 条对照规则时，
    这条用例是**假的**：7 个非法重写里 5 个从没被检查过，而它照样是绿的。
    """
    hi = float(p["expectations.json"]["derived"]["safe_window"]["hi"])
    entries = _illegal_entries(p)
    bad = []
    for tol in _tolerances(p):
        if tol >= hi:
            continue
        for name, accepted_at, _ in entries:
            if _at(accepted_at, tol):
                bad.append((name, tol))
    n_below = sum(1 for t in _tolerances(p) if t < hi)
    return (not bad,
            f"窗口上界 {hi:.4e} 以下共 {n_below} 级容差 × {len(entries)} 个非法重写"
            f"（{sum(1 for r in _illegal(p))} 条对照规则 + "
            f"{len(entries) - sum(1 for r in _illegal(p))} 档 ε 变体），"
            f"被错误接受的（重写, 容差）对：{bad}",
            {"window_hi": hi, "wrongly_accepted": bad,
             "n_illegal": len(entries), "n_tolerances_below": n_below})


@_case("legal_rewrites_are_all_accepted_at_or_above_the_window")
def _i_legal_above(p):
    """**窗口下界及以上**的每一级容差，都必须接受**全部**合法重写。

    与上一条互补：只断言「非法被拒」的话，一个「什么都拒」的校验器也能通过。
    两条一起才说明窗口**两边都紧**。
    """
    lo = float(p["expectations.json"]["derived"]["safe_window"]["lo"])
    bad = []
    for tol in _tolerances(p):
        if tol < lo:
            continue
        for r in _legal(p):
            if not _at(r["accepted_at"], tol):
                bad.append((r["rule"], tol))
    return (not bad,
            f"窗口下界 {lo:.4e} 及以上共 "
            f"{sum(1 for t in _tolerances(p) if t >= lo)} 级容差，"
            f"被错误拒掉的（规则, 容差）对：{bad}",
            {"window_lo": lo, "wrongly_rejected": bad})


@_case("epsilon_scan_is_proportional")
def _i_eps(p):
    """`bias_epsilon(ε)` 的漂移必须正比于 ε：`drift / ε` 近似是常数。

    红的情况：ε 没真的注进图里（例如注到了副本上）——
    那样漂移与 ε 无关，「容差崖可以解析定位」这条结论就不成立。

    ⚠️ **容差不能拍脑袋。** 第一版写 `spread < 1e-9`，实测 6.42e-05 ⇒ 红。
    但这不是实现错了，是**浮点相消**：ε=1e-12 的扰动加在量级为 1 的张量上，
    只保得住约 12 位十进制 ⇒ `drift` 的相对误差 ~ `2.2e-16 / ε`。
    所以容差取 **`4 × 机器精度 / min(ε)`**（4 倍余量）：
    这个界是**推出来的**，不是试出来的，而且它随 ε 一起变 ——
    真把 ε 注丢了（drift 与 ε 无关）时 spread 会大到这个界之外。
    """
    import sys
    per: dict[float, float] = {}
    for v in p["adversarial.json"]["variants"]:
        # ⚠️ 变体名是 `bias_epsilon(1e-12)` 这种**带参数**的形式，
        #    不是 `bias_epsilon`。第一版用 `==` 匹配，于是扫到 0 个点。
        if str(v["variant"]).startswith("bias_epsilon") and v.get("epsilon"):
            per[float(v["epsilon"])] = float(v["drift"]) / float(v["epsilon"])
    if len(per) < 2:
        return False, f"ε 扫描只有 {len(per)} 个点，算不出比例常数", per
    vals = list(per.values())
    spread = (max(vals) - min(vals)) / max(vals)
    bound = 4.0 * sys.float_info.epsilon / min(per)
    declared = float(p["adversarial.json"]["drift_per_epsilon"])
    ok = (spread <= bound
          and all(abs(v - declared) <= bound * declared for v in vals))
    return (ok,
            f"drift/ε 的 {len(vals)} 个读数在 [{min(vals):.6f}, {max(vals):.6f}]，"
            f"相对散布 {spread:.2e}；浮点相消给出的界 {bound:.2e}"
            f"（= 4ε_machine/min(ε)）；声明的系数 {declared:.6f}",
            {"drift_per_epsilon": per, "spread": spread, "bound": bound,
             "declared": declared})


@_case("epsilon_grid_spans_the_cliff")
def _i_eps_grid(p):
    """ε 网格必须在崖的**两侧都有格子**，否则这条「崖」根本没被测到。

    ⚠️ 判据**从 `variants` 现算**，不读 `eps_grid_spans_cliff` 那个字段 ——
    第一版那个字段在 `adversarial.py` 里是**写死的 `True`**：
    它看起来像一个测量结果，其实是一个假设，而读者没法从产物内部发现。
    现在字段是算出来的，这条用例负责**独立复核**它。
    """
    adv = p["adversarial.json"]
    eps_rows = [v for v in adv["variants"] if v.get("epsilon") is not None]
    n_acc = sum(1 for v in eps_rows if v["accepted_any"])
    n_eps = len(eps_rows)
    spans = 0 < n_acc < n_eps
    declared = bool(adv["eps_grid_spans_cliff"])
    ok = spans and declared and int(adv["eps_rows_accepted"]) == n_acc \
        and int(adv["eps_rows_total"]) == n_eps
    return (ok,
            f"{n_eps} 档 ε 里有 {n_acc} 档在某一级容差下被接受"
            f"（0 或全部 ⇒ 网格全在崖的一侧）；产物声明 {declared}、"
            f"记录 {adv['eps_rows_accepted']}/{adv['eps_rows_total']}",
            {"n_eps": n_eps, "n_accepted": n_acc, "spans": spans,
             "declared": declared})


# ===================================================================== Pareto 与前沿
@_case("frontier_members_are_pairwise_nondominated")
def _i_front_nondom(p):
    """前沿内部任意两点必须**互不支配**。

    红的情况：前沿是「按分数取前 k 个」而不是按支配关系算的 ——
    那样前沿里会混进被别的成员支配的点，而它看起来仍然是一条前沿。
    """
    bad = []
    for r in _rungs(p):
        for name, cell in sorted(r["cells"].items()):
            pts = _trial_objectives(cell)
            idx = non_dominated_indices(pts)
            if len(idx) != len(pts):
                bad.append(f"{r['strategy']}@{r['tolerance']:g}/{name}")
    return (not bad,
            f"20 级 × 5 搜索器里，前沿成员存在互相支配的格子：{bad}",
            {"bad_cells": bad})


@_case("frontier_members_are_feasible_at_their_rung")
def _i_front_feas(p):
    """前沿成员在自己的那一级上必须**可行**。

    红的情况：前沿把不可行点也收进去了（`reject` 下不可能，
    但 `penalty` 下如果按分数挑就会）—— 那样「交出去的答案可行」这句话就空了。
    """
    bad = []
    for r, cell in _cells(p):
        for t in cell.get("frontier", []):
            if not t["feasible"]:
                bad.append((r["strategy"], r["tolerance"], cell["searcher"],
                            t["plan_id"]))
    return (not bad,
            f"前沿里不可行的成员：{bad[:6]}（共 {len(bad)} 个）",
            {"n_infeasible": len(bad), "examples": bad[:6]})


@_case("frontier_members_drift_within_the_rung_tolerance")
def _i_front_drift(p):
    """前沿成员的漂移必须 ≤ 它所在那一级的容差。

    红的情况：前沿是跨级合并出来的（例如把 τ=1e-3 的点混进了 τ=0 的级）——
    那样「τ=0 的前沿」这句话是假的，而数字仍然漂亮。
    """
    bad = []
    for r, cell in _cells(p):
        tol = float(r["tolerance"])
        for t in cell.get("frontier", []):
            if float(t["drift"]) > tol + 1e-30:
                bad.append((r["strategy"], tol, t["plan_id"], t["drift"]))
    return (not bad,
            f"漂移超过本级容差的前沿成员：{bad[:6]}（共 {len(bad)} 个）",
            {"n_over": len(bad), "examples": bad[:6]})


@_case("frontier_counts_are_consistent")
def _i_front_counts(p):
    """`n_frontier` / `n_frontier_distinct` / `len(frontier)` 三个数必须自洽。

    红的情况：`n_frontier_distinct` 忘记去重（数成计划数）——
    那样「12 个计划只有 2 个权衡点」这条结论会被抹平，
    而门槛正好落在它上面 ⇒ 门槛会靠重复点通过。
    """
    bad = []
    for r, cell in _cells(p):
        pts = _trial_objectives(cell)
        distinct = len({tuple(round(x, 12) for x in o) for o in pts})
        if not (int(cell["n_frontier"]) == len(pts)
                and int(cell["n_frontier_distinct"]) == distinct
                and distinct <= len(pts)):
            bad.append(f"{r['strategy']}@{r['tolerance']:g}/{cell['searcher']}"
                       f"(n_frontier={cell['n_frontier']}, len={len(pts)}, "
                       f"n_distinct={cell['n_frontier_distinct']}, 实数={distinct})")
    return (not bad,
            f"三个数不自洽的格子：{bad[:6]}（共 {len(bad)} 个）",
            {"bad_cells": bad[:6], "n_bad": len(bad)})


@_case("hypervolume_recomputes_from_the_frontier")
def _i_hv_recompute(p):
    """超体积必须能**从产物里的前沿与参考点重算出来**。

    红的情况：超体积用了另一套参考点或另一套点集 ——
    那样「前沿 A 比 B 宽」这类结论全部建立在两个不同的尺子上。
    """
    ref = p["space.json"]["reference"]
    bad = []
    for key, block in sorted(p["ladder.json"]["frontier"].items()):
        pts = [list(t["objectives"]) for t in block["points"]]
        got = hypervolume(pts, ref)
        if abs(got - float(block["hypervolume"])) > 1e-9:
            bad.append((key, got, block["hypervolume"]))
    return (not bad,
            f"重算超体积与产物的差 > 1e-9 的级：{bad}",
            {"bad": bad, "reference": ref})


@_case("hypervolume_reference_point_is_dominated")
def _i_hv_ref(p):
    """参考点必须被前沿里**每一个**点支配。

    红的情况：参考点没取到「比所有点都差」—— 那样「多一个烂点，超体积反而变大」，
    而那个变化会被误读成「搜索变好了」。
    """
    ref = tuple(p["space.json"]["reference"])
    bad = []
    for key, block in sorted(p["ladder.json"]["frontier"].items()):
        for t in block["points"]:
            if not dominates(tuple(t["objectives"]), ref):
                bad.append((key, t["plan_id"]))
    return (not bad,
            f"不被参考点支配的前沿成员：{bad[:6]}（共 {len(bad)} 个）；参考点 {ref}",
            {"n_not_dominating": len(bad), "examples": bad[:6]})


@_case("hypervolume_is_monotone_in_the_frontier_set")
def _i_hv_mono(p):
    """超体积必须对「加一个被支配的点」不变、对「加一个支配性的点」不减。

    红的情况：超体积实现把重复点重复计了（实测出现过 12 个前沿点只有 2 个不同）——
    那样「前沿变宽」会被重复点伪造出来。
    """
    ref = tuple(p["space.json"]["reference"])
    block = p["ladder.json"]["frontier"]["reject@0"]
    base = [list(t["objectives"]) for t in block["points"]]
    hv0 = hypervolume(base, ref)
    # 被支配的点：三个目标都取前沿里最小的那个再减一点
    mins = [min(o[i] for o in base) for i in range(3)]
    dominated_point = [mins[0] - 0.01, mins[1] - 0.01, mins[2] - 0.01]
    hv_dom = hypervolume(base + [dominated_point], ref)
    # 支配性的点：三个目标都取最大的那个再加一点
    maxs = [max(o[i] for o in base) for i in range(3)]
    better = [min(1.0, maxs[0] + 0.05), min(1.0, maxs[1] + 0.05),
              min(1.0, maxs[2] + 0.05)]
    hv_better = hypervolume(base + [better], ref)
    ok = (abs(hv_dom - hv0) < 1e-12 and hv_better >= hv0 - 1e-12)
    return (ok,
            f"加被支配点：{hv0:.9f} → {hv_dom:.9f}（要求不变）；"
            f"加支配点：{hv0:.9f} → {hv_better:.9f}（要求不减）",
            {"hv0": hv0, "hv_with_dominated": hv_dom, "hv_with_better": hv_better})


@_case("dominates_is_antisymmetric")
def _i_antisym(p):
    """支配关系不能双向成立。

    红的情况：`dominates` 用「≥ 且有一个 >」写成「≥」——
    双向成立会让非支配排序死循环，或把全部点分到同一层（前沿退化成全体）。
    """
    pts = [list(t["objectives"]) for t in
           p["ladder.json"]["frontier"]["reject@0"]["points"]]
    pts += [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.5, 0.0, 1.0], [0.5, 0.0, 1.0]]
    bad = [(i, j) for i, a in enumerate(pts) for j, b in enumerate(pts)
           if i != j and dominates(a, b) and dominates(b, a)]
    return (not bad,
            f"{len(pts)} 个点两两检查，双向支配的（i, j）对：{bad[:6]}"
            f"（共 {len(bad)} 对）",
            {"n_pairs": len(bad), "examples": bad[:6]})


@_case("dominates_is_transitive")
def _i_trans(p):
    """支配必须传递：`a≻b 且 b≻c ⇒ a≻c`。

    红的情况：比较用了「有一个目标相等就算支配」这种近似 ——
    传递性一破，非支配排序的结果就依赖点的**输入顺序**，
    而「前沿是确定的」这句话就不成立了。
    """
    base = [list(t["objectives"]) for t in
            p["ladder.json"]["frontier"]["reject@0"]["points"]]
    grid = [[i / 4, j / 4, k / 4] for i in range(5) for j in range(5)
            for k in range(5)]
    pts = base + grid
    bad = []
    for a in pts:
        for b in pts:
            if not dominates(a, b):
                continue
            for c in pts:
                if dominates(b, c) and not dominates(a, c):
                    bad.append((a, b, c))
                    if len(bad) >= 3:
                        break
            if len(bad) >= 3:
                break
        if len(bad) >= 3:
            break
    return (not bad,
            f"{len(pts)} 个点上抽查传递性，反例：{bad}",
            {"n_counterexamples": len(bad), "examples": bad})


@_case("hv_gap_column_matches_the_hypervolumes")
def _i_hv_gap(p):
    """`hv_gap_vs_oracle` 必须**就是** `1 − hv / hv_oracle`。

    红的情况：缺口列另算（例如用「最好均值」的差）——
    那样同一张表里两个对象用了两个口径，而它们看起来可以互相验证。
    """
    bad = []
    for r, cell in _cells(p):
        orc = r["cells"]["oracle"]["hypervolume"]
        if not orc:
            continue
        want = 1.0 - float(cell["hypervolume"]) / float(orc)
        got = float(cell["hv_gap_vs_oracle"])
        if abs(want - got) > 1e-9:
            bad.append(f"{r['strategy']}@{r['tolerance']:g}/{cell['searcher']}"
                       f"(报 {got:.9f}, 应为 {want:.9f})")
    return (not bad,
            f"缺口列与超体积对不上的格子：{bad[:6]}（共 {len(bad)} 个）",
            {"n_bad": len(bad), "examples": bad[:6]})


@_case("best_answer_is_a_frontier_member")
def _i_best_on_front(p):
    """交出去的答案必须**在前沿上**。

    红的情况：答案按「加权均值」单独挑（本项目确实用 `mean_gain` 排序）——
    如果它落在前沿之外，说明交付的是一个**被支配**的点，
    而报告里写着「交付物是前沿」。
    """
    best = p["ladder.json"]["best_answer"]
    pts = p["ladder.json"]["frontier"]["reject@0"]["points"]
    ids = {t["plan_id"] for t in pts}
    ok = best["plan_id"] in ids
    return (ok,
            f"头条答案 {best['plan_id']} {'在' if ok else '**不在**'} τ=0 前沿的 "
            f"{len(ids)} 个成员里",
            {"best": best["plan_id"], "n_frontier": len(ids)})


@_case("objectives_are_in_unit_range")
def _i_obj_range(p):
    """三个目标都必须落在 `[0, 1]`。

    红的情况：归一化除以了「最好值」而不是「基线值」——
    那样目标会超过 1，而三目标超体积的参考点 `[−…, −…, −…]` 就失去意义。
    """
    bad = []
    for r, cell in _cells(p):
        for t in list(cell.get("frontier", [])) + \
                ([cell["best"]] if cell.get("best") else []) + \
                ([cell["best_score"]] if cell.get("best_score") else []):
            for i, v in enumerate(t["objectives"]):
                if not (0.0 - 1e-12 <= float(v) <= 1.0 + 1e-12):
                    bad.append((r["strategy"], r["tolerance"], cell["searcher"],
                                t["plan_id"], i, v))
    return (not bad,
            f"越界的目标读数：{bad[:6]}（共 {len(bad)} 个）",
            {"n_out": len(bad), "examples": bad[:6]})


@_case("mean_gain_is_the_mean_of_the_objectives")
def _i_mean(p):
    """凡是报了 `mean_gain` 的地方，它必须**就是**三个目标的算术平均。

    红的情况：均值用了别的权重（那正是「结果层不许标量化」要防的事）——
    同一份产物里两个数用了两个口径，读者没法从内部发现。

    ⚠️ 判据要限定在「**报了**这个字段的地方」：格子里的 trial 不带 `mean_gain`
    （那是**前沿视图**才有的字段），第一版对所有 trial 硬取这个键，
    于是用例自己抛 `KeyError` —— 用例自己抛异常也算失败（不许吞）。
    """
    bad = []
    seen = 0
    for key, block in sorted(p["ladder.json"]["frontier"].items()):
        for t in block["points"]:
            if "mean_gain" not in t:
                continue
            seen += 1
            if abs(mean_gain(t["objectives"]) - float(t["mean_gain"])) > 1e-9:
                bad.append((key, t["plan_id"]))
    ba = p["ladder.json"]["best_answer"]
    seen += 1
    if abs(mean_gain(ba["objectives"]) - float(ba["mean_gain"])) > 1e-9:
        bad.append(("best_answer", ba["plan_id"]))
    return (not bad and seen > 0,
            f"检查了 {seen} 个带 mean_gain 的条目，均值与三目标平均对不上的："
            f"{bad[:6]}（共 {len(bad)} 个）",
            {"n_checked": seen, "n_bad": len(bad), "examples": bad[:6]})


@_case("frontier_views_agree")
def _i_views(p):
    """同一级前沿有**两个视图**：`rung.cells[*].frontier`（trial 全字段）
    与 `ladder.frontier[key].points`（精简点）。两者必须描述同一批计划。

    红的情况：两个视图各算各的（本项目实测它们**字段集就不同**：
    trial 17 个键、point 9 个键）—— 一旦哪天其中一个多留了一个点或少算一次
    去重，两个视图会静默分叉，而它们都被印在报告里。
    """
    bad = []
    for key, block in sorted(p["ladder.json"]["frontier"].items()):
        strategy, _, tol_s = key.partition("@")
        r = _rung(p, strategy, float(tol_s))
        for name, cell in sorted(r["cells"].items()):
            a = {t["plan_id"]: tuple(round(x, 12) for x in t["objectives"])
                 for t in cell.get("frontier", [])}
            b = {t["plan_id"]: tuple(round(x, 12) for x in t["objectives"])
                 for t in block["points"]}
            # 只有 oracle 的点集等于整条前沿（其余搜索器是子集）
            if name == "oracle" and a != b:
                bad.append((key, name, len(a), len(b)))
    return (not bad,
            f"两个前沿视图对不上的级：{bad[:6]}（共 {len(bad)} 个）",
            {"n_bad": len(bad), "examples": bad[:6]})


# ===================================================================== 搜索器与预算
@_case("searchers_respect_their_budget")
def _i_budget(p):
    """任何格子的**唯一提案数**都不能超过预算；oracle 的预算是整个空间。

    红的情况：预算记账被缓存命中污染（本项目真踩过：`oracle` 先跑填满缓存，
    后面每个搜索器的每次提案都算命中 ⇒ `used` 恒为 0）——
    那样「预算 120」这个口径失效，而三张对照表都是按它读的。
    """
    size = int(p["space.json"]["space_size"])
    per = int(p["search.json"]["budget_per_searcher"])
    bad = []
    for r, cell in _cells(p):
        want = size if cell["searcher"] == "oracle" else per
        if int(cell["budget"]) != want or int(cell["used"]) > want:
            bad.append((r["strategy"], r["tolerance"], cell["searcher"],
                        cell["budget"], cell["used"]))
    return (not bad,
            f"预算或用量越界的格子：{bad[:6]}（共 {len(bad)} 个）；"
            f"oracle 预算 {size}、其余 {per}",
            {"n_bad": len(bad), "examples": bad[:6]})


@_case("proposal_accounting_adds_up")
def _i_accounting(p):
    """提案记账必须自洽：`n_unique ≤ used ≤ n_proposed`，且
    `hits == n_proposed − used`（自己重复提案的次数）。

    红的情况：`n_proposed` 数了缓存命中（或 `used` 忘了计）——
    那样「搜索器 A 比 B 省预算」这类结论是账面假象。
    """
    bad = []
    for r, cell in _cells(p):
        np_, used = int(cell["n_proposed"]), int(cell["used"])
        uniq, hits = int(cell["n_unique"]), int(cell["hits"])
        if not (uniq <= used <= np_ and hits == np_ - used
                and int(cell["shared"]) >= 0):
            bad.append(f"{r['strategy']}@{r['tolerance']:g}/{cell['searcher']}"
                       f"(proposed={np_}, used={used}, unique={uniq}, hits={hits}, "
                       f"shared={cell['shared']})")
    return (not bad,
            f"记账不自洽的格子：{bad[:6]}（共 {len(bad)} 个）",
            {"n_bad": len(bad), "examples": bad[:6]})


@_case("feasible_count_never_exceeds_the_feasible_space")
def _i_feasible_count(p):
    """一个格子里的**可行候选数**不能超过**这一级的可行域大小**。

    本项目真踩过：提案不去重 ⇒ `evolution` 报出 `n_feasible = 171`，
    而 τ=0 的可行域只有 85 个计划。**171 > 85 是一个自相矛盾的数字**，
    但它不报错、也不影响前沿（重复点彼此非支配）——只让「评估了多少候选」变成假数。
    """
    bad = []
    for r, cell in _cells(p):
        space = int(r["n_feasible_space"])
        if int(cell["n_feasible"]) > space:
            bad.append(f"{r['strategy']}@{r['tolerance']:g}/{cell['searcher']}"
                       f"(n_feasible={cell['n_feasible']} > 可行域 {space})")
    return (not bad,
            f"可行候选数超过可行域的格子：{bad[:6]}（共 {len(bad)} 个）",
            {"n_bad": len(bad), "examples": bad[:6]})


@_case("budget_sweep_agrees_with_the_main_rung")
def _i_sweep(p):
    """预算扫描里「主预算」那一行，必须与主阶梯 τ=0 的读数**逐位相同**。

    红的情况：扫描用了别的参考点 / 别的种子 ——
    那样「策略与预算的贡献分不开」这条结论里的两半来自两个实验，
    而它们被画在同一张表里，看起来可比。
    """
    per = int(p["search.json"]["budget_per_searcher"])
    row = next((r for r in p["budget_sweep.json"]["rows"]
                if int(r["budget"]) == per), None)
    if row is None:
        return False, f"预算扫描里没有主预算 {per} 这一行", None
    main = _rung(p, "reject", _tau0(p))["cells"]
    bad = []
    for name in sorted(main):
        a = float(main[name]["hypervolume"])
        b = float(row["cells"][name]["hypervolume"])
        if abs(a - b) > 1e-12:
            bad.append((name, a, b))
    return (not bad,
            f"主预算 {per} 的扫描行与主阶梯 τ=0 对不上的搜索器：{bad}",
            {"budget": per, "mismatch": bad})


@_case("budget_asymmetry_is_declared")
def _i_asymmetry(p):
    """oracle 的预算 = 空间大小，其余 = `budget_per_searcher`，且**必须显式声明**。

    红的情况：把 oracle 也压到同一个预算 ——
    那样它交出的只是「120 个里的最优」，`oracle_is_optimal` 从数学必然退化成可能，
    而报告里仍然写着「oracle」。
    """
    a = p["search.json"]["budget_asymmetry"]
    size = int(p["space.json"]["space_size"])
    per = int(p["search.json"]["budget_per_searcher"])
    ok = int(a["oracle"]) == size and int(a["others"]) == per and bool(a["note"])
    return (ok,
            f"声明的预算不对称：oracle={a['oracle']}（空间 {size}）、"
            f"其余={a['others']}（配置 {per}）；理由字段{'非空' if a['note'] else '为空'}",
            {"oracle": a["oracle"], "others": a["others"],
             "space_size": size, "per": per})


@_case("ladder_tolerances_are_strictly_increasing")
def _i_tol_order(p):
    """阶梯容差必须严格递增，且从 0 起。

    红的情况：容差列表里有重复或倒序 ——
    `_rung(strategy, tol)` 的查找会命中错误的一级，
    而「阶梯」这个词就不再指向一条有序的曲线。
    """
    tols = _tolerances(p)
    ok = (tols[0] == 0.0
          and all(b > a for a, b in zip(tols, tols[1:])))
    return (ok, f"容差序列 {tols}",
            {"tolerances": tols, "first_is_zero": tols[0] == 0.0})


@_case("ladder_has_both_strategies_for_every_tolerance")
def _i_ladder_shape(p):
    """每一级容差都必须有 `reject` 与 `penalty` **两种**策略的格子。

    红的情况：某一级只跑了一种策略 ——
    「软硬约束给出同一个答案」这类对照结论里，有一半的格子是空的，
    而它在报告里看起来像「两边一样」。
    """
    tols = _tolerances(p)
    want = {(s, t) for s in ("reject", "penalty") for t in tols}
    got = {(r["strategy"], r["tolerance"]) for r in _rungs(p)}
    ok = want == got
    return (ok,
            f"应有 {len(want)} 级（2 策略 × {len(tols)} 容差），实有 {len(got)} 级；"
            f"缺 {sorted(want - got)[:4]}，多 {sorted(got - want)[:4]}",
            {"n_expected": len(want), "n_actual": len(got)})


@_case("monotonicity_is_consistent_with_the_rungs")
def _i_mono(p):
    """单调性读数的长度必须与阶梯级数一致，且 `monotone` 与违反列表一致。

    红的情况：单调性只覆盖了部分级（例如跳过了 0 那一级）——
    那样「阶梯单调不减」这句话没有覆盖它该覆盖的东西，而它永远是绿的。
    """
    tols = _tolerances(p)
    bad = []
    for m in p["ladder.json"]["monotonicity"]:
        if (list(m["tolerances"]) != tols
                or len(m["hypervolume"]) != len(tols)
                or len(m["n_feasible_space"]) != len(tols)
                or len(m["n_frontier"]) != len(tols)
                or bool(m["monotone"]) != (not m["hypervolume_violations"])):
            bad.append(m["strategy"])
    return (not bad,
            f"单调性读数与阶梯不一致的策略：{bad}（共 {len(p['ladder.json']['monotonicity'])} 条读数）",
            {"bad": bad, "n_rungs": len(_rungs(p))})


@_case("cost_model_is_monotone_in_launch_overhead")
def _i_cost(p):
    """`launch_overhead_us` 越大，基线与最优延迟都必须不减。

    红的情况：成本模型把 launch 开销加到了错的一侧（例如从最优延迟里减掉）——
    那样「融合买到的是访存而不是 launch」这条结论的**方向**就是反的。
    """
    cs = sorted(p["benchmark.json"]["cost_sweep"],
                key=lambda c: float(c["launch_overhead_us"]))
    base = [float(c["baseline_latency_us"]) for c in cs]
    best = [float(c["best_latency_us"]) for c in cs]
    sp = [float(c["speedup"]) for c in cs]
    ok = (all(b >= a for a, b in zip(base, base[1:]))
          and all(b >= a for a, b in zip(best, best[1:]))
          and all(b >= a for a, b in zip(sp, sp[1:])))
    return (ok,
            f"{len(cs)} 档 launch_overhead：基线延迟 {base}；最好延迟 {best}；"
            f"speedup {[round(x, 6) for x in sp]}",
            {"baseline": base, "best": best, "speedup": sp})


# ===================================================================== 产物与报告
@_case("versioned_and_unversioned_cover_every_product")
def _i_version_cover(p):
    """`VERSIONED_FILES ∪ UNVERSIONED_FILES` 必须**恰好等于**全部产物。

    红的情况：加了一个新产物却忘了登记 ——
    它会**既不进版本摘要、也不进豁免名单**，
    于是「两个版本号相同 ⇒ 连产物都相同」这句话对它是假的，而它看起来没问题。

    ⚠️ `report.md` / `report.html` / `version.json` 由 `emit.py` 渲染，
    在这一步（`finish`）还没进内存 ⇒ 它们必须显式豁免，
    否则这条用例会永远红，而「永远红的用例」与「没有用例」一样没用。
    """
    from .pipeline import UNVERSIONED_FILES, VERSIONED_FILES
    rendered = {"report.md", "report.html", "version.json"}
    declared = set(VERSIONED_FILES) | set(UNVERSIONED_FILES)
    in_memory = set(p) - {"version.json"}
    missing = sorted(in_memory - declared)          # 产出了但没登记
    not_yet = sorted(declared - in_memory)          # 登记了但这一步还没产出
    ok = not missing and set(not_yet) <= rendered
    return (ok,
            f"登记的产物 {len(declared)} 个、`finish` 时内存里 {len(in_memory)} 个；"
            f"**产出了但没登记**的 {missing}；登记了但此刻还没渲染的 {not_yet}"
            f"（后者只许是 {sorted(rendered)}）",
            {"missing": missing, "not_yet": not_yet,
             "n_versioned": len(VERSIONED_FILES),
             "n_unversioned": len(UNVERSIONED_FILES)})


@_case("summary_is_a_projection_of_other_products")
def _i_summary(p):
    """`summary.json` 的每个字段都必须**能对回**别的产物。

    红的情况：摘要另算了一遍（例如自己重算超体积）——
    那样摘要与明细会静默分叉，而摘要是人最先看的那一份。
    """
    s, sp, lad, adv, lin, acc = (p["summary.json"], p["space.json"],
                                 p["ladder.json"], p["adversarial.json"],
                                 p["lineage.json"], p["acceptance.json"])
    r0 = lad["frontier"]["reject@0"]
    bad = []
    checks = [
        ("space_size", s["space_size"], sp["space_size"]),
        ("n_distinct_digests", s["n_distinct_digests"], sp["n_distinct_digests"]),
        ("hypervolume_at_tau0", s["hypervolume_at_tau0"], r0["hypervolume"]),
        ("frontier_points_at_tau0", s["frontier_points_at_tau0"],
         len(r0["points"])),
        ("best_answer", s["best_answer"], lad["best_answer"]),
        ("protocol_digest", s["protocol_digest"],
         p["config.json"]["protocol_digest"]),
        ("safe_window", s["safe_window"],
         p["expectations.json"]["derived"]["safe_window"]),
        ("illegal_accepted_at_default_tolerance",
         s["illegal_accepted_at_default_tolerance"],
         adv["n_accepted_at_default_tolerance"]),
        ("n_legal_bit_identical", s["n_legal_bit_identical"],
         sum(1 for r in lin["rules"] if r["kind"] == "legal" and r["bit_identical"])),
        ("acceptance.n", s["acceptance"]["n"], acc["n"]),
        ("acceptance.n_passed", s["acceptance"]["n_passed"], acc["n_passed"]),
        ("acceptance.n_failed", s["acceptance"]["n_failed"], acc["n_failed"]),
    ]
    for name, a, b in checks:
        if a != b:
            bad.append((name, a, b))
    return (not bad,
            f"摘要与明细对不上的字段：{bad}",
            {"n_fields": len(checks), "mismatch": bad})


@_case("expectation_verdicts_match_expect_and_observed")
def _i_verdict(p):
    """`verdict` 必须**就是** `expect` 与 `observed` 的比较结果。

    红的情况：裁定与读数各写各的（README 手抄那一版就是这么错的）——
    那时报告里会出现「原期望 是 / 实测 是 / 裁定 被推翻」这种自相矛盾的行，
    而**汇总数可能仍然对得上**（两行互换 ⇒ 成立数与推翻数不变）。
    """
    bad = []
    for it in p["expectations.json"]["items"]:
        want = "held" if bool(it["expect"]) == bool(it["observed"]) else "refuted"
        if it["verdict"] != want:
            bad.append((it["name"], it["expect"], it["observed"], it["verdict"]))
    n = len(p["expectations.json"]["items"])
    held = sum(1 for i in p["expectations.json"]["items"] if i["verdict"] == "held")
    return (not bad,
            f"{n} 条期望里裁定与（原期望, 实测）不符的：{bad}；"
            f"重新数出来 {held} 成立 / {n - held} 被推翻"
            f"（产物报 {p['expectations.json']['n_held']} / {p['expectations.json']['n_refuted']}）",
            {"n_bad": len(bad), "examples": bad[:6], "recounted_held": held})


@_case("expectation_claims_are_complete")
def _i_claims(p):
    """每条期望都要有**人话声明**，且声明的条数与求值函数一一对应。

    红的情况：加了一个求值函数却忘了写 `claim` ——
    报告里会出现「没有结论的一行」，而它看起来像一条通过的期望。
    """
    exp = p["expectations.json"]
    empty = [i["name"] for i in exp["items"] if not str(i.get("claim", "")).strip()]
    ok = (not empty and not exp["orphan_evaluators"]
          and not exp["missing_evaluators"])
    return (ok,
            f"{len(exp['items'])} 条期望；缺 claim 的 {empty}；"
            f"孤儿求值函数 {exp['orphan_evaluators']}；"
            f"缺求值函数 {exp['missing_evaluators']}",
            {"n_items": len(exp["items"]), "empty_claims": empty})


@_case("timing_expectations_are_out_of_the_versioned_products")
def _i_timing(p):
    """依赖实测耗时的期望**不许**留在进版本摘要的产物里。

    本项目真踩过：时序期望的证据里嵌着微秒数，
    于是「同一份输入跑两次得到同一个版本号」当场失效。
    红的情况：有人把 `TIMING_EVALUATORS` 里的东西放回了 `expectations.json`。
    """
    exp = p["expectations.json"]
    in_versioned = [i["name"] for i in exp["items"] if i.get("timing_dependent")]
    timing = p["benchmark.json"].get("timing_expectations", [])
    ok = (not in_versioned) and len(timing) == int(exp["n_timing"]) > 0
    return (ok,
            f"`expectations.json` 里的时序期望 {in_versioned}；"
            f"`benchmark.json` 里 {len(timing)} 条（声明 {exp['n_timing']} 条）",
            {"in_versioned": in_versioned, "n_timing": len(timing),
             "declared": exp["n_timing"]})


@_case("op_counts_sum_to_the_node_count")
def _i_opcounts(p):
    """算子计数之和必须等于节点数。

    红的情况：图里混进了没被计数的节点（例如别名节点）——
    那样「49 个节点」与 `op_counts` 就是两套账，而两份数都印在报告里。
    """
    b = p["baseline.json"]
    total = sum(int(v) for v in b["op_counts"].values())
    ok = total == int(b["n_nodes"])
    return (ok,
            f"op_counts {b['op_counts']} 之和 = {total}，n_nodes = {b['n_nodes']}",
            {"sum": total, "n_nodes": b["n_nodes"], "op_counts": b["op_counts"]})


@_case("baseline_and_space_digest_counts_agree")
def _i_digest_agree(p):
    """空间里「不同的图指纹数」必须等于血缘图的节点数。

    两个数从两条不同的路径算出来（一个扫空间、一个建图）——
    对不上说明有一条路径漏了去重或漏了节点。
    """
    a = int(p["space.json"]["n_distinct_digests"])
    b = int(p["lineage.json"]["graph"]["n_nodes"])
    ok = a == b
    return (ok, f"空间 {a} 张不同的图 vs 血缘图 {b} 个节点", {"space": a, "lineage": b})


@_case("order_sensitivity_counts_are_internally_consistent")
def _i_order(p):
    """顺序敏感性的三个计数必须自洽：有序对 = 2 × 无序对，
    且敏感的有序对 = 2 × 敏感的无序对，例子数不少于敏感对数。

    红的情况：只数了「A→B」忘了反向 ——
    那样「顺序重要」这句话的强度被低估一半，而它看起来仍然是一条结论。
    """
    o = p["lineage.json"]["order_sensitivity"]
    ok = (int(o["n_ordered_pairs"]) == 2 * int(o["n_unordered_pairs"])
          and int(o["n_order_sensitive_ordered"])
          == 2 * int(o["n_order_sensitive_unordered"])
          and len(o["examples"]) >= int(o["n_order_sensitive_unordered"]))
    return (ok,
            f"有序 {o['n_ordered_pairs']}（应 = 2×{o['n_unordered_pairs']}）；"
            f"敏感有序 {o['n_order_sensitive_ordered']}"
            f"（应 = 2×{o['n_order_sensitive_unordered']}）；"
            f"例子 {len(o['examples'])} 条",
            {"ordered": o["n_ordered_pairs"], "unordered": o["n_unordered_pairs"],
             "sensitive_ordered": o["n_order_sensitive_ordered"],
             "sensitive_unordered": o["n_order_sensitive_unordered"],
             "n_examples": len(o["examples"])})


@_case("graph_edges_reference_known_nodes")
def _i_edges(p):
    """血缘图里每条边的两端都必须是已存在的节点。

    红的情况：节点集合按「图指纹」去重、边按「(from, to)」记录，
    但某一端用了未登记的 id —— 那样图会**静默缺节点**，
    而节点数正好是那条门槛（`min_lineage_nodes`）量的东西。
    """
    g = p["lineage.json"]["graph"]
    nodes = set(g["nodes"])
    bad = [(e["from"], e["to"]) for e in g["edges"]
           if e["from"] not in nodes or e["to"] not in nodes]
    return (not bad,
            f"{len(g['edges'])} 条边、{len(nodes)} 个节点；悬空的边：{bad[:6]}"
            f"（共 {len(bad)} 条）",
            {"n_edges": len(g["edges"]), "n_nodes": len(nodes),
             "dangling": bad[:6]})


@_case("baseline_is_its_own_reference_point")
def _i_baseline_zero(p):
    """基线在三个目标上必须**恰好**是 0 —— 它定义了「下降比例」的分母。

    红的情况：基线指标被别的东西改过（例如成本模型换了单位）——
    那样「延迟下降 31.88%」这句话的**分母**变了，而它看起来仍然是个百分比。
    """
    b = p["baseline.json"]
    keys = ("bytes_moved", "const_nodes", "flops", "latency_us", "launches",
            "nodes", "peak_bytes")
    bad = [k for k in keys if not (isinstance(b["metrics"][k], (int, float))
                                   and math.isfinite(float(b["metrics"][k])))]
    lo = p["ladder.json"]["best_answer"]
    # 头条答案的目标必须 = (基线 − 最好) / 基线（逐目标重算）
    m = b["metrics"]
    want = [
        (m["latency_us"] - lo["latency_us"]) / m["latency_us"],
        (m["peak_bytes"] - lo["peak_bytes"]) / m["peak_bytes"],
        (m["launches"] - lo["launches"]) / m["launches"],
    ]
    diff = [abs(a - b2) for a, b2 in zip(want, lo["objectives"])]
    ok = not bad and max(diff) < 1e-9
    return (ok,
            f"基线的 7 个指标都可算：{not bad}；"
            f"头条答案的目标重算差 {[f'{d:.2e}' for d in diff]}",
            {"metrics_ok": not bad, "objective_diffs": diff,
             "recomputed": want, "declared": lo["objectives"]})


# --------------------------------------------------------------------- 主入口
def run_cases(products: dict[str, Any]) -> dict[str, Any]:
    """跑完全部对抗用例。**用例自己抛异常也算失败**（不许吞）。"""
    cases: list[dict[str, Any]] = []
    for name in sorted(_CASES):
        try:
            ok, detail, observed = _CASES[name](products)
        except Exception as exc:                       # noqa: BLE001
            ok, detail, observed = (
                False,
                f"**用例自己抛异常**：{type(exc).__name__}: {exc} "
                f"—— 一个抛异常的用例与一条红的用例，都说明这里有东西不对",
                None)
        cases.append({"name": name, "ok": bool(ok), "detail": detail,
                      "observed": observed})
    failed = [c["name"] for c in cases if not c["ok"]]
    return {
        "cases": cases,
        "n_cases": len(cases),
        "n_cases_passed": sum(1 for c in cases if c["ok"]),
        "n_cases_failed": len(failed),
        "failed_cases": failed,
        "note": ("对抗用例是**不可调**的不变量断言：配置里没有它们的阈值。"
                 "红了说明代码或产物自相矛盾，不是「这次没达标」。"
                 "⚠️ `adversarial.json` 里那 7 个**非法变体**是被测量的"
                 "实验对象（数据），不是对抗用例。"),
    }


def registered_cases() -> tuple[str, ...]:
    """用例名（供接线检查与门槛使用 —— 不跑用例就能知道有多少条）。"""
    return tuple(sorted(_CASES))
