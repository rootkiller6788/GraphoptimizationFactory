"""验收：门槛 vs 实测。**每一条都必须能红。**

## 三条写门槛的纪律（本项目真的踩过）

1. **一条恒真的判据与没有判据，在输出里长得一模一样。**
   第一版写过 `max_illegal_acceptance_below_cliff = 0` ——
   「非法变体在低于自己漂移的容差下被接受」。按定义不可能发生（`drift <= τ` 才被接受），
   所以它永远是绿的。已删。写门槛时问：**它在什么情况下会红？**
2. **一条从不触发的保险丝与没有保险丝无法区分。**
   所以保险丝自己也要有读数：`min_acceptance_checks` 断言验收表至少有这么长 ——
   删掉几条门槛会红，而不是让报告看起来更干净。
3. **凡「有它在兜着」的量，都要算出来并断言。**
   `every_threshold_is_wired` 用 AST 扫源码，确认配置里每个门槛键
   **真的被某段代码读过**。一个写在配置里、没人读的门槛，
   与一个不存在的门槛在输出里完全一样 —— 而它看起来像一道防线。

## `every_threshold_is_wired` 的实现口径

- 扫**除 `config.py` 之外**的全部 `.py`：`DEFAULTS` 里的键本身就是字符串字面量，
  把 `config.py` 算进来，这条检查会**恒真**（每个键都在自己的定义处出现一次）。
- **排除文档字符串**：设计意图就写在文档字符串里，把「解释」当成「接线」是自欺。
  实现上用 `ast` 节点身份排除（不是文本匹配）—— 同一个字符串既当文档字符串
  又当真字面量时，文本匹配会误判。
- 判据是「**加一个门槛要不要改这个文件**」，不是「这个文件里有没有字符串」。
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Any, Callable, Sequence

from .ladder import mean_gain

#: 门槛键 → 求值函数。返回 `(observed, passed, detail)`。
_CHECKS: dict[str, Callable[[dict[str, Any], Any], tuple[Any, bool, str]]] = {}


def _check(name: str):
    def deco(fn):
        _CHECKS[name] = fn
        return fn
    return deco


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _reject_cells(p: dict[str, Any], tol: float) -> list[dict[str, Any]]:
    for r in p["ladder.json"]["rungs"]:
        if r["strategy"] == "reject" and r["tolerance"] == tol:
            return [r["cells"][n] for n in sorted(r["cells"])]
    return []


def _tau0(p: dict[str, Any]) -> float:
    return min(float(x) for x in p["config.json"]["resolved"]["ladder"]["tolerances"])


# --------------------------------------------------------------------- 门槛
@_check("min_ladder_rungs")
def _c_rungs(p, thr):
    n = len(p["config.json"]["resolved"]["ladder"]["tolerances"])
    return n, n >= int(thr), f"阶梯 {n} 级（要求 ≥ {thr}）"


@_check("max_structural_failures")
def _c_struct(p, thr):
    d = p["space.json"]["evaluator"]
    n = int(d["n_structural"])
    return n, n <= int(thr), (f"结构失败 {n} 次（要求 ≤ {thr}）"
                              f"；原因分布 {d['structural_reasons']}")


@_check("min_lineage_nodes")
def _c_lin(p, thr):
    n = int(p["lineage.json"]["graph"]["n_nodes"])
    return n, n >= int(thr), f"血缘图 {n} 个不同图指纹（要求 ≥ {thr}）"


@_check("min_distinct_plans")
def _c_distinct(p, thr):
    n = int(p["space.json"]["n_distinct_digests"])
    return n, n >= int(thr), f"空间产生 {n} 张不同的图（要求 ≥ {thr}）"


@_check("min_order_sensitive_pairs")
def _c_order(p, thr):
    n = int(p["lineage.json"]["order_sensitivity"]["n_order_sensitive_unordered"])
    return n, n >= int(thr), f"{n} 个无序规则对换序后结果不同（要求 ≥ {thr}）"


@_check("min_bit_identical_rewrites")
def _c_bit(p, thr):
    n = sum(1 for r in p["lineage.json"]["rules"]
            if r["kind"] == "legal" and r["bit_identical"])
    return n, n >= int(thr), f"{n} 条合法重写逐位等价（要求 ≥ {thr}）"


@_check("min_inexact_rewrites")
def _c_inexact(p, thr):
    n = sum(1 for r in p["lineage.json"]["rules"]
            if r["kind"] == "legal" and not r["bit_identical"])
    return n, n >= int(thr), f"{n} 条合法重写不逐位等价（要求 ≥ {thr}）"


@_check("safe_window_nonempty")
def _c_window(p, thr):
    w = p["expectations.json"]["derived"]["safe_window"]
    return (bool(w["nonempty"]), bool(w["nonempty"]) == bool(thr),
            f"安全窗口 {'非空' if w['nonempty'] else '为空'}"
            f"（{w['lo']:.3e} → {w['hi']:.3e}）")


@_check("min_frontier_points")
def _c_front(p, thr):
    """前沿的**有效大小**：不同的目标向量个数。

    按「计划数」量会被顺序冗余灌水 —— 实测：12 个前沿计划只有 **2** 个
    不同的权衡点，其余 10 个是「换个顺序得到同一张图」。
    两个数都报出来（`n_frontier` 与 `n_frontier_distinct`），
    门槛落在后者上，因为前者可以靠重复点通过。
    """
    key = f"reject@{_tau0(p):g}"
    rung = next(r for r in p["ladder.json"]["rungs"]
                if r["strategy"] == "reject" and r["tolerance"] == _tau0(p))
    o = rung["cells"]["oracle"]
    n = int(o["n_frontier_distinct"])
    return (n, n >= int(thr),
            f"τ=0 前沿 {o['n_frontier']} 个计划 / **{n} 个不同权衡点**"
            f"（要求 ≥ {thr}）")


@_check("max_greedy_hv_gap_vs_oracle")
def _c_greedy(p, thr):
    c = next(c for c in _reject_cells(p, _tau0(p)) if c["searcher"] == "greedy")
    g = c["hv_gap_vs_oracle"]
    if not _finite(g):
        return None, False, "greedy 的超体积缺口算不出来"
    return g, g <= float(thr), f"greedy 超体积缺口 {g:.6f}（要求 ≤ {thr}）"


@_check("min_searcher_hv_gain_over_random_at_tight_budget")
def _c_beat(p, thr):
    """在**紧预算**下，最好的非随机搜索器相对随机的超体积增益。

    为什么必须限定在紧预算：实测主预算 120 下随机搜索已经饱和
    （拿到 oracle 的全部超体积），在那里比策略等于比运气。
    10~30 次才是策略起作用的地方 —— 这个门槛把「哪个预算下比较」这个前提
    写进了配置，而不是留给读者猜。
    """
    sw = p["budget_sweep.json"]
    tight = int(sw["tight_budget"])
    row = next((m for m in sw["margins"] if m["budget"] == tight), None)
    if row is None:
        return None, False, f"预算扫描里没有紧预算 {tight}"
    g = row["gain_over_random"]
    return (g, g >= float(thr),
            f"紧预算 {tight} 下最好的非随机搜索器 = {row['best_searcher']}，"
            f"相对随机增益 {g:+.6f}（要求 ≥ {thr}）")


@_check("min_ladder_feasible_steps")
def _c_feas_steps(p, thr):
    """阶梯上**可行域严格扩张**的级数。

    为什么不量「质量上升」：实测质量那一列**全平**
    （10 级容差、前沿 12 点、超体积 0.054010148327 一格没动）——
    放宽容差在这里只买到更大的可行域，买不到更好的答案。
    ⇒ 「阶梯动没动」只能量在可行域上，而它是能红的。
    """
    rows = p["ladder.json"]["monotonicity"]
    n = max(r["feasible_steps"] for r in rows)
    detail = "；".join(
        f"{r['strategy']} 可行域 {' → '.join(str(x) for x in r['n_feasible_space'])}"
        for r in rows)
    return n, n >= int(thr), f"最多 {n} 步可行域扩张（要求 ≥ {thr}）；{detail}"


@_check("oracle_is_optimal")
def _c_oracle(p, thr):
    cells = _reject_cells(p, _tau0(p))
    orc = next(c for c in cells if c["searcher"] == "oracle")
    og = mean_gain(orc["best"]["objectives"]) if orc["best"] else float("nan")
    worse = []
    for c in cells:
        if c["searcher"] == "oracle" or c["best"] is None:
            continue
        g = mean_gain(c["best"]["objectives"])
        if g > og + 1e-12:
            worse.append((c["searcher"], g))
    return (bool(worse), (not worse) == bool(thr),
            f"oracle 最好均值 {og:.6f}；超过它的搜索器 {worse}")


@_check("max_answer_infeasible_rate")
def _c_answer(p, thr):
    cells = _reject_cells(p, _tau0(p))
    n_front = sum(c["n_frontier"] for c in cells)
    bad = sum(1 for c in cells for t in c["frontier"] if not t["feasible"])
    rate = (bad / n_front) if n_front else 0.0
    return rate, rate <= float(thr), (f"前沿成员不可行率 {rate:.6f}"
                                      f"（{bad}/{n_front}，要求 ≤ {thr}）")


@_check("min_monotone_strategies")
def _c_mono(p, thr):
    rows = p["ladder.json"]["monotonicity"]
    n = sum(1 for r in rows if r["monotone"])
    return n, n >= int(thr), f"{n}/{len(rows)} 种策略的 oracle 阶梯单调不减（要求 ≥ {thr}）"


@_check("min_adversarial_cases")
def _c_cases(p, thr):
    """**对抗用例条数的保险丝** —— 删用例会让它红。

    它读的是 `invariants._CASES` 的**注册表大小**，不跑用例：
    于是它能在 `acceptance.json` 的 `cases` 被填进去**之前**就生效
    （`cases` 依赖 `summary.json`，而 `summary.json` 依赖验收结果 ⇒ 不能循环）。

    ⚠️ 口径：这里数的**不是** `adversarial.json` 里那 7 个非法变体。
    那些是**被测量的实验对象**（数据）；「对抗用例」只指不变量断言。
    第一版把两者混称，于是报告里写「7 条对抗用例」，而 L5 各项目是 24~38 条 ——
    **同一个词指了两个不同的东西，读者没法从报告内部发现。**
    """
    from .invariants import registered_cases
    n = len(registered_cases())
    return (n, n >= int(thr),
            f"注册了 {n} 条对抗用例（要求 ≥ {thr}）—— 它们是不可调的不变量断言，"
            f"与 `adversarial.json` 里那 7 个**非法变体**不是一回事")


@_check("min_acceptance_checks")
def _c_self(p, thr):
    """**保险丝的自指读数**：验收表本身的长度。

    它不需要额外数据 —— 它的输入就是「有多少条门槛」。
    这正是它作为保险丝的意义：删门槛会让它红。

    ⚠️ 它**不能**去读 `acceptance.json`（那正是正在构造的产物，
    自指会读到半成品）。所以它数配置里的键，再加 1 ——
    加的那 1 是 `every_threshold_is_wired`，它不在配置里（它是结构性不变量，
    不是可调门槛，因此不许被调松）。
    """
    n = len(p["config.json"]["resolved"]["acceptance"]) + 2
    return n, n >= int(thr), f"验收表共 {n} 条（含本条与两条常开保险丝，要求 ≥ {thr}）"


# --------------------------------------------------------------------- 接线检查
def string_literals(pkg_dir: Path, exclude: Sequence[str] = ()) -> set[str]:
    """包内全部 `.py` 的字符串字面量，**排除文档字符串与 `exclude` 里的文件**。"""
    out: set[str] = set()
    excluded = set(exclude)
    for path in sorted(pkg_dir.rglob("*.py")):
        if path.name in excluded:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        doc_nodes: set[int] = set()
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if not isinstance(body, list) or not body:
                continue
            first = body[0]
            if (isinstance(first, ast.Expr)
                    and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                doc_nodes.add(id(first.value))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and id(node) not in doc_nodes):
                out.add(node.value)
    return out


def wiring_report(cfg: dict[str, Any], pkg_dir: Path) -> dict[str, Any]:
    """三类接线核对，每类都可能红。"""
    lits = string_literals(pkg_dir, exclude=("config.py",))

    acc_keys = sorted(cfg["acceptance"])
    unwired = [k for k in acc_keys if k not in lits]

    # 有求值函数、但配置里没声明（写了代码忘了写配置）
    exp_declared = set(cfg["expectations"])
    from .expectations import _EVALUATORS
    exp_orphan = sorted(set(_EVALUATORS) - exp_declared)
    exp_missing = sorted(exp_declared - set(_EVALUATORS))

    # 搜索器声明消费的搜索超参必须真的存在于 config.search
    from .search import declared_params
    declared = declared_params()
    search_keys = set(cfg["search"])
    bad_params = sorted({f"{s}.{k}" for s, ks in declared.items()
                         for k in ks if k not in search_keys})

    return {
        "n_acceptance_keys": len(acc_keys),
        "unwired_acceptance_keys": unwired,
        "orphan_expectation_evaluators": exp_orphan,
        "missing_expectation_evaluators": exp_missing,
        "searcher_params_not_in_config": bad_params,
        "all_wired": not unwired and not exp_orphan and not exp_missing
                     and not bad_params,
        "note": "扫的是除 config.py 之外的全部源码；文档字符串不计入接线。",
    }


# --------------------------------------------------------------------- 主入口
def run_acceptance(products: dict[str, Any], pkg_dir: Path | None = None
                   ) -> dict[str, Any]:
    cfg = products["config.json"]["resolved"]
    declared = cfg["acceptance"]

    checks: list[dict[str, Any]] = []
    for name in sorted(declared):
        thr = declared[name]
        fn = _CHECKS.get(name)
        if fn is None:
            checks.append({
                "name": name, "threshold": thr, "observed": None, "passed": False,
                "detail": "**没有任何检查函数认领这条门槛** —— 它看起来像防线，其实不是",
            })
            continue
        observed, passed, detail = fn(products, thr)
        checks.append({"name": name, "threshold": thr,
                       "observed": (None if observed is None
                                    else (round(float(observed), 12)
                                          if isinstance(observed, (int, float))
                                          and not isinstance(observed, bool)
                                          else observed)),
                       "passed": bool(passed), "detail": detail})

    orphan = sorted(set(_CHECKS) - set(declared))
    if pkg_dir is None:
        pkg_dir = Path(__file__).resolve().parent
    wiring = wiring_report(cfg, pkg_dir)
    checks.append({
        "name": "every_threshold_is_wired", "threshold": True,
        "observed": wiring["all_wired"], "passed": bool(wiring["all_wired"]),
        "detail": ("全部门槛键都被源码读过、全部期望都有求值函数、"
                   "搜索器声明的超参都在配置里"
                   if wiring["all_wired"] else
                   f"未接线：{wiring['unwired_acceptance_keys']}；"
                   f"孤儿期望：{wiring['orphan_expectation_evaluators']}；"
                   f"缺求值函数：{wiring['missing_expectation_evaluators']}；"
                   f"声明了但配置里没有的超参：{wiring['searcher_params_not_in_config']}"),
    })

    # 常开保险丝：参考点必须整次运行唯一。
    # 第一版按级取参考点，于是「放宽容差」同时移动了参考点，
    # 超体积凭空涨了 68% —— **那 68% 全部是假象**，而它看起来像一条漂亮结论。
    refs = {tuple(r["reference"]) for r in products["ladder.json"]["rungs"]}
    ref_ok = len(refs) == 1
    checks.append({
        "name": "reference_is_shared_across_rungs", "threshold": 1,
        "observed": len(refs), "passed": ref_ok,
        "detail": (f"全部 {len(products['ladder.json']['rungs'])} 级共用同一个参考点 "
                   f"{sorted(refs)[0]}" if ref_ok else
                   f"出现了 {len(refs)} 个不同的参考点：{sorted(refs)}"
                   " —— 各用各的参考点，超体积之间没法比"),
    })

    return {
        "checks": checks,
        "n": len(checks),
        "n_passed": sum(1 for c in checks if c["passed"]),
        "n_failed": sum(1 for c in checks if not c["passed"]),
        "failed": [c["name"] for c in checks if not c["passed"]],
        "orphan_check_functions": orphan,
        "wiring": wiring,
    }
