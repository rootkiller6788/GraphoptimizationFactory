"""设计期望：**先写下我相信什么，再让实测来说话。**

## 为什么这一节是本项目最值钱的部分

验收门槛回答「做对了吗」，设计期望回答「**我原来想错了什么**」。
后者只有事先写下来才成立 —— 事后从数据里挑一条漂亮的当「期望」，那是编故事。

所以每条期望在 `configs/default.json` 里就写死了 `claim` 与 `expect`，
本模块只负责**把它算出来**，并给出三种裁定：

| verdict | 含义 |
| --- | --- |
| `held` | 实测与期望一致 |
| `refuted` | **实测推翻了期望** —— 这是产出，不是失败 |
| `unmeasurable` | 这个量在本次配置下算不出来（**不许当成通过**） |

`unmeasurable` 单独一档是刻意的：把「算不出来」并进「held」或「refuted」
都会让报告撒谎。缺失的证据 ≠ 否定的证据（附录 C 184）。

## `derived`：从实测里长出来、但不属于任何一条期望的数

- `safe_window`：`[max(合法漂移), min(非法漂移))` —— **容差该取多少**的答案。
  它非空，就说明存在一个容差区间能把全部合法重写收下、把全部非法重写挡住。
- `cliff_by_variant`：每个非法变体在哪一级容差第一次被收下。

这两个数是本项目对「容差」这个设计轴的全部回答，**它们不依赖任何期望**。
"""

from __future__ import annotations

import math
from typing import Any, Callable

#: 期望名 → 求值函数。函数只吃产物字典（**不读运行期对象**）。
_EVALUATORS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {}

#: **依赖实测耗时的期望**。它们的读数随机器变，所以
#: **不许进 `expectations.json`**（那是进版本摘要的产物）——
#: 实测抓到过：`analytic_latency_tracks_wallclock` 的证据里嵌着
#: 每次求值的微秒数，于是「同一份输入跑两次」得到了两个版本号。
#: 它们落进 `benchmark.json["timing_expectations"]`，与其它耗时读数在一起。
TIMING_EVALUATORS: frozenset[str] = frozenset({"analytic_latency_tracks_wallclock"})


def _register(name: str):
    def deco(fn):
        _EVALUATORS[name] = fn
        return fn
    return deco


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _rung(products: dict[str, Any], strategy: str, tol: float) -> dict[str, Any]:
    return products["ladder.json"]["frontier"].get(f"{strategy}@{tol:g}", {})


def _cell(products: dict[str, Any], strategy: str, tol: float,
          searcher: str) -> dict[str, Any] | None:
    for r in products["ladder.json"]["rungs"]:
        if r["strategy"] == strategy and r["tolerance"] == tol:
            return r["cells"].get(searcher)
    return None


def safe_window(products: dict[str, Any]) -> dict[str, Any]:
    """安全容差窗口：`[max 合法漂移, min 非法漂移)`。"""
    rules = products["lineage.json"]["rules"]
    legal = [r for r in rules if r["kind"] == "legal" and _finite(r["drift"])]
    illegal = [r["drift"] for r in products["adversarial.json"]["variants"]
               if _finite(r["drift"])]
    if not legal or not illegal:
        return {"nonempty": False, "lo": None, "hi": None,
                "reason": "合法或非法变体为空，窗口无从定义"}
    lo = max(r["drift"] for r in legal)
    hi = min(illegal)
    lo_rule = max(legal, key=lambda r: r["drift"])["rule"]
    return {
        "nonempty": lo < hi,
        "lo": lo,
        "hi": hi,
        "lo_from": lo_rule,
        "decades": (math.log10(hi / lo) if lo > 0 and hi > 0 else None),
        "default_tolerance": products["adversarial.json"]["default_tolerance"],
        "default_inside": lo <= products["adversarial.json"]["default_tolerance"] < hi,
    }


def _illegal_at_default(products: dict[str, Any]) -> int:
    return int(products["adversarial.json"]["n_accepted_at_default_tolerance"])


# --------------------------------------------------------------------- 各条期望
@_register("exact_equality_accepts_all_legal_rewrites")
def _e_exact(p: dict[str, Any]) -> dict[str, Any]:
    legal = [r for r in p["lineage.json"]["rules"] if r["kind"] == "legal"]
    acc = [r["rule"] for r in legal if r["accepted_at"].get("0.0", False)]
    return {
        "observed": len(acc) == len(legal),
        "detail": f"τ=0 接受了 {len(acc)}/{len(legal)} 条合法重写",
        "evidence": {"accepted": acc,
                     "rejected": [r["rule"] for r in legal if r["rule"] not in acc]},
    }


@_register("fusion_is_bit_identical")
def _e_fusion(p: dict[str, Any]) -> dict[str, Any]:
    row = next((r for r in p["lineage.json"]["rules"]
                if r["rule"] == "operator_fusion"), None)
    if row is None:
        return {"observed": None, "detail": "operator_fusion 不在规则表里",
                "evidence": {}}
    return {
        "observed": bool(row["bit_identical"]),
        "detail": f"operator_fusion 漂移 = {row['drift']:.3e}，"
                  f"命中 {row['hits']} 处，节点 {row['delta']['nodes']:+d}",
        "evidence": {"drift": row["drift"], "hits": row["hits"],
                     "delta": row["delta"]},
    }


@_register("order_matters")
def _e_order(p: dict[str, Any]) -> dict[str, Any]:
    o = p["lineage.json"]["order_sensitivity"]
    return {
        "observed": o["n_order_sensitive_unordered"] >= 1,
        "detail": f"{o['n_order_sensitive_unordered']}/{o['n_unordered_pairs']} "
                  f"个无序规则对换序后结果不同",
        "evidence": {"n_unordered_pairs": o["n_unordered_pairs"],
                     "examples": o["examples"][:3]},
    }


@_register("greedy_is_myopic")
def _e_greedy(p: dict[str, Any]) -> dict[str, Any]:
    g = _cell(p, "reject", 0.0, "greedy")
    o = _cell(p, "reject", 0.0, "oracle")
    if g is None or o is None or not _finite(g["hv_gap_vs_oracle"]):
        return {"observed": None, "detail": "缺 greedy/oracle 读数", "evidence": {}}
    return {
        "observed": g["hv_gap_vs_oracle"] > 1e-9,
        "detail": f"greedy 超体积缺口 {g['hv_gap_vs_oracle']:.6f}"
                  f"（用 {g['used']} 次评估 vs oracle {o['used']} 次），"
                  f"前沿 {g['n_frontier']} 点 vs oracle {o['n_frontier']} 点",
        "evidence": {"hv_gap": g["hv_gap_vs_oracle"],
                     "greedy_used": g["used"], "oracle_used": o["used"],
                     "greedy_frontier": g["n_frontier"],
                     "oracle_frontier": o["n_frontier"]},
    }


@_register("ladder_is_monotone")
def _e_mono(p: dict[str, Any]) -> dict[str, Any]:
    rows = p["ladder.json"]["monotonicity"]
    ok = all(r["monotone"] for r in rows)
    return {
        "observed": ok,
        "detail": "；".join(
            f"{r['strategy']} {'单调不减' if r['monotone'] else '出现下降'}"
            f"（{len(r['violations'])} 处违反）" for r in rows),
        "evidence": {"per_strategy": rows},
    }


@_register("safe_tolerance_window_is_nonempty")
def _e_window(p: dict[str, Any]) -> dict[str, Any]:
    w = safe_window(p)
    return {
        "observed": bool(w["nonempty"]),
        "detail": (f"安全窗口 [{w['lo']:.3e}, {w['hi']:.3e})"
                   f"，宽 {w['decades']:.2f} 个数量级"
                   if w["nonempty"] else w.get("reason", "窗口为空")),
        "evidence": w,
    }


@_register("launch_overhead_dominates")
def _e_launch(p: dict[str, Any]) -> dict[str, Any]:
    sweep = p["benchmark.json"].get("cost_sweep")
    if not sweep:
        return {"observed": None, "detail": "没有成本扫描", "evidence": {}}
    by_ov = {row["launch_overhead_us"]: row["speedup"] for row in sweep}
    lo, hi = min(by_ov), max(by_ov)
    if not _finite(by_ov[hi]) or by_ov[hi] <= 0:
        return {"observed": None, "detail": "最大 launch 开销下 speedup 非正",
                "evidence": {"sweep": sweep}}
    share = (by_ov[hi] - by_ov[lo]) / by_ov[hi]
    return {
        "observed": share >= 0.5,
        "detail": f"launch 开销 {lo:g}→{hi:g} µs 时 speedup "
                  f"{by_ov[lo]:.4f}→{by_ov[hi]:.4f}，"
                  f"**launch 贡献占比 {share:.4f}**",
        "evidence": {"share_from_launch": share, "sweep": sweep},
    }


@_register("objectives_are_independent")
def _e_obj_indep(p: dict[str, Any]) -> dict[str, Any]:
    corr = p["space.json"]["objective_correlation"]
    if not corr:
        return {"observed": None, "detail": "相关矩阵算不出来（可行解太少）",
                "evidence": {}}
    worst, pair = 0.0, ("", "")
    for a, row in corr.items():
        for b, v in row.items():
            if a == b or not _finite(v):
                continue
            if abs(v) > abs(worst):
                worst, pair = v, (a, b)
    return {
        "observed": abs(worst) < 0.95,
        "detail": f"最强相关 {pair[0]} ~ {pair[1]}：{worst:+.6f}"
                  f"（|corr| {'<' if abs(worst) < 0.95 else '>='} 0.95）",
        "evidence": {"matrix": corr, "worst_pair": list(pair), "worst": worst},
    }


@_register("peak_memory_is_an_independent_axis")
def _e_peak(p: dict[str, Any]) -> dict[str, Any]:
    corr = p["space.json"]["objective_correlation"]
    v = (corr.get("memory_saving", {}) or {}).get("speedup_gain")
    if not _finite(v):
        return {"observed": None, "detail": "该相关系数算不出来", "evidence": {}}
    return {
        "observed": abs(v) < 0.95,
        "detail": f"memory_saving ~ speedup_gain = {v:+.6f}",
        "evidence": {"corr": v},
    }


@_register("searchers_beat_random")
def _e_beat_random(p: dict[str, Any]) -> dict[str, Any]:
    cells = {name: _cell(p, "reject", 0.0, name)
             for name in p["search.json"]["searchers"]}
    if any(c is None for c in cells.values()):
        return {"observed": None, "detail": "缺搜索器读数", "evidence": {}}
    base = cells["random"]["hypervolume"]
    worse = [n for n, c in cells.items()
             if n not in ("random",) and c["hypervolume"] < base - 1e-12]
    return {
        "observed": not worse,
        "detail": "；".join(f"{n} HV={c['hypervolume']:.6f}"
                            for n, c in sorted(cells.items())),
        "evidence": {"random_hv": base, "worse_than_random": worse,
                     "hv": {n: c["hypervolume"] for n, c in sorted(cells.items())}},
    }


@_register("analytic_latency_tracks_wallclock")
def _e_analytic(p: dict[str, Any]) -> dict[str, Any]:
    w = p["benchmark.json"].get("wallclock", {})
    rho = w.get("spearman_rho")
    if not _finite(rho):
        return {"observed": None, "detail": f"秩相关算不出来（{w.get('n_plans')} 个样本）",
                "evidence": {"n_plans": w.get("n_plans")}}
    return {
        "observed": bool(rho >= 0.5),
        "detail": f"解析延迟 vs 实测耗时的 Spearman ρ = {rho:+.6f}"
                  f"（{w['n_plans']} 个前沿计划，重复取最小）",
        "evidence": {"rho": rho, "n_plans": w["n_plans"], "rows": w["rows"]},
    }


@_register("validator_catches_all_illegal")
def _e_validator(p: dict[str, Any]) -> dict[str, Any]:
    n = _illegal_at_default(p)
    tol = p["adversarial.json"]["default_tolerance"]
    return {
        "observed": n == 0,
        "detail": f"默认容差 τ={tol:g} 下漏掉 {n}/"
                  f"{p['adversarial.json']['n_illegal_variants']} 个非法变体："
                  f"{p['adversarial.json']['accepted_at_default_tolerance']}",
        "evidence": {"n_accepted": n, "tolerance": tol,
                     "accepted": p["adversarial.json"]["accepted_at_default_tolerance"]},
    }


@_register("penalty_keeps_answers_feasible")
def _e_penalty(p: dict[str, Any]) -> dict[str, Any]:
    """口径：**「交出去的答案」= 搜索器评分最高的那个候选**（`best_score`）。

    这是搜索器**自己以为**的最优。软约束下它会跑到不可行区域去，
    于是「交出去的答案仍然可行」这句话是否成立，取决于你问的是
    「按可行性挑的答案」还是「按评分挑的答案」——**这两个是不同的点**，
    所以两个都要报（`best` 与 `best_score` 在产物里都留着）。
    """
    cells: list[dict[str, Any]] = []
    for rung in p["ladder.json"]["rungs"]:
        if rung["strategy"] != "penalty":
            continue
        for name in p["search.json"]["searchers"]:
            c = rung["cells"].get(name)
            if c is not None:
                cells.append(c)
    bad = [(c["searcher"], c["tolerance"])
           for c in cells if c["best_score"] is not None
           and not c["best_score"]["feasible"]]
    n_best_infeasible = sum(1 for c in cells if c["best"] is None)
    return {
        "observed": not bad,
        "detail": f"penalty 策略下 {len(bad)}/{len(cells)} 个格子的"
                  f"**评分最优候选不可行**（按可行性挑的答案不可行数 = "
                  f"{n_best_infeasible}）",
        "evidence": {"n_cells": len(cells), "n_best_score_infeasible": len(bad),
                     "examples": bad[:6],
                     "n_best_infeasible": n_best_infeasible},
    }


def _sweep_row(p: dict[str, Any], budget: int) -> dict[str, Any] | None:
    sw = p.get("budget_sweep.json")
    if not sw:
        return None
    return next((r for r in sw["rows"] if r["budget"] == budget), None)


def _sweep_margin(p: dict[str, Any], budget: int) -> dict[str, Any] | None:
    sw = p.get("budget_sweep.json")
    if not sw:
        return None
    return next((m for m in sw["margins"] if m["budget"] == budget), None)


@_register("searchers_beat_random_at_tight_budget")
def _e_beat_tight(p: dict[str, Any]) -> dict[str, Any]:
    tight = int(p["budget_sweep.json"]["tight_budget"])
    m = _sweep_margin(p, tight)
    if m is None:
        return {"observed": None, "detail": f"扫描里没有紧预算 {tight}", "evidence": {}}
    return {
        "observed": m["gain_over_random"] > 0.0,
        "detail": f"紧预算 {tight}：最好的非随机搜索器 {m['best_searcher']} "
                  f"超体积 {m['best_hypervolume']:.6f} vs 随机 "
                  f"{m['random_hypervolume']:.6f}，增益 {m['gain_over_random']:+.6f}",
        "evidence": m,
    }


@_register("random_saturates_at_main_budget")
def _e_saturate(p: dict[str, Any]) -> dict[str, Any]:
    main = int(p["config.json"]["resolved"]["search"]["budget_per_searcher"])
    r = _sweep_row(p, main)
    if r is None:
        return {"observed": None, "detail": f"扫描里没有主预算 {main}", "evidence": {}}
    gap = r["cells"]["random"]["hv_gap_vs_oracle"]
    sat = p["budget_sweep.json"]["random_saturation_budget"]
    return {
        "observed": bool(gap is not None and gap <= 1e-9),
        "detail": f"主预算 {main} 下随机的超体积缺口 {gap:+.3e}"
                  f"（随机第一次饱和的预算是 {sat}）",
        "evidence": {"gap": gap, "saturation_budget": sat, "main_budget": main},
    }


@_register("soft_constraint_changes_the_answer")
def _e_soft(p: dict[str, Any]) -> dict[str, Any]:
    """软约束 vs 硬约束：**逐格比「交出去的答案」**。

    预期会被推翻：本项目的漂移尺度（合法重写 ~1e-16）配 `drift_scale=1.0`，
    惩罚项小到不影响排序 ⇒ 两种策略几乎给出同一个答案。
    **这不是 bug，是「软约束的软硬取决于违反量的量纲」这条规律的读数。**
    """
    rungs = p["ladder.json"]["rungs"]
    searchers = p["search.json"]["searchers"]
    diffs: list[dict[str, Any]] = []
    total = 0
    for tol in sorted({r["tolerance"] for r in rungs}):
        a = next((r for r in rungs if r["strategy"] == "reject"
                  and r["tolerance"] == tol), None)
        b = next((r for r in rungs if r["strategy"] == "penalty"
                  and r["tolerance"] == tol), None)
        if a is None or b is None:
            continue
        for n in searchers:
            ca, cb = a["cells"].get(n), b["cells"].get(n)
            if ca is None or cb is None:
                continue
            total += 1
            ra = (ca["best"] or {}).get("rules")
            rb = (cb["best"] or {}).get("rules")
            hv_same = abs(ca["hypervolume"] - cb["hypervolume"]) < 1e-15
            if ra != rb or not hv_same:
                diffs.append({"tolerance": tol, "searcher": n,
                              "reject": ra, "penalty": rb, "hv_same": hv_same})
    return {
        "observed": bool(diffs),
        "detail": f"{len(diffs)}/{total} 个格子两种策略给出的答案不同"
                  + (f"；例如 τ={diffs[0]['tolerance']:g} 的 {diffs[0]['searcher']}"
                     if diffs else "（**软硬约束在本项目的漂移尺度下等价**）"),
        "evidence": {"n_cells": total, "n_different": len(diffs),
                     "examples": diffs[:6],
                     "drift_scale": p["config.json"]["resolved"]["constraints"]["drift_scale"]},
    }


@_register("ladder_improves_quality")
def _e_ladder_quality(p: dict[str, Any]) -> dict[str, Any]:
    """放宽容差到底买到了什么。

    实测（本项目）：**什么都没买到**。可行域从 85 涨到 400，
    但前沿始终是同样的 12 个点、超体积 0.054010148327 一格没动、
    最优点高度也一格没动。

    ⇒ 语义等价这条硬约束在这里**完全不 binding**：
    那 3 条不逐位等价的重写（`reassociate_add` / `matmul_reassociate` /
    `scale_into_weight`）**全部被逐位等价的规则支配** ——
    它们省不下任何东西，只是引入了误差。
    ⇒ 「容差该取多少」的答案不是「权衡后的折中」，而是
    **「取安全窗口里最小的那个，零代价」**。
    """
    rows = p["ladder.json"]["monotonicity"]
    active = max(r["active_steps"] for r in rows)
    feas = max(r["feasible_steps"] for r in rows)
    hv0 = rows[0]["hypervolume"][0]
    hv1 = rows[0]["hypervolume"][-1]
    return {
        "observed": active > 0,
        "detail": f"超体积 {hv0:.12f} → {hv1:.12f}（{active} 步上升 / "
                  f"{rows[0]['flat_steps']} 步平坦），"
                  f"可行域扩张 {feas} 步 —— "
                  + ("**质量随容差改善**" if active > 0
                     else "**放宽容差买不到任何质量**：前沿与最优点都一格没动"),
        "evidence": {"active_steps": active, "feasible_steps": feas,
                     "hypervolume_first": hv0, "hypervolume_last": hv1,
                     "n_feasible_space": rows[0]["n_feasible_space"],
                     "n_frontier": rows[0]["n_frontier"]},
    }


@_register("ladder_has_flat_top")
def _e_flat(p: dict[str, Any]) -> dict[str, Any]:
    rows = p["ladder.json"]["monotonicity"]
    detail = "；".join(
        f"{r['strategy']}：{r['active_steps']} 步上升、{r['flat_steps']} 步平坦"
        for r in rows)
    # 判据：至少有一半的级是平坦的（= 阶梯在高容差端用完）
    ok = all(r["flat_steps"] >= (len(r["tolerances"]) - 1) / 2 for r in rows)
    return {"observed": ok, "detail": detail,
            "evidence": {"per_strategy": [{"strategy": r["strategy"],
                                           "active": r["active_steps"],
                                           "flat": r["flat_steps"]} for r in rows]}}


# --------------------------------------------------------------------- 主入口
def run_expectations(products: dict[str, Any]) -> dict[str, Any]:
    declared = products["config.json"]["resolved"]["expectations"]
    items: list[dict[str, Any]] = []
    timing_items: list[dict[str, Any]] = []
    missing: list[str] = []

    for name in sorted(declared):
        spec = declared[name]
        fn = _EVALUATORS.get(name)
        if fn is None:
            missing.append(name)
            items.append({
                "name": name, "claim": spec["claim"], "expect": bool(spec["expect"]),
                "observed": None, "verdict": "unmeasurable",
                "detail": "**没有任何求值函数认领这条期望**", "evidence": {},
            })
            continue
        r = fn(products)
        observed = r["observed"]
        if observed is None:
            verdict = "unmeasurable"
        elif bool(observed) == bool(spec["expect"]):
            verdict = "held"
        else:
            verdict = "refuted"
        row = {
            "name": name,
            "claim": spec["claim"],
            "expect": bool(spec["expect"]),
            "observed": None if observed is None else bool(observed),
            "verdict": verdict,
            "detail": r["detail"],
            "evidence": r.get("evidence", {}),
            "timing_dependent": name in TIMING_EVALUATORS,
        }
        (timing_items if name in TIMING_EVALUATORS else items).append(row)

    # 反向核对：有求值函数、但配置里没声明的期望（写代码写了、忘了写进配置）
    orphan = sorted(set(_EVALUATORS) - set(declared))

    return {
        "items": items,
        "timing_items": timing_items,
        "derived": {"safe_window": safe_window(products)},
        "n": len(items),
        "n_held": sum(1 for i in items if i["verdict"] == "held"),
        "n_refuted": sum(1 for i in items if i["verdict"] == "refuted"),
        "n_unmeasurable": sum(1 for i in items if i["verdict"] == "unmeasurable"),
        "n_timing": len(timing_items),
        "missing_evaluators": missing,
        "orphan_evaluators": orphan,
        "note": "时序类期望（TIMING_EVALUATORS）不在本文件里，它们在 "
                "benchmark.json['timing_expectations'] —— 它们的读数随机器变，"
                "进了本文件就会让「同一份输入两次得到同一个版本号」失效。",
    }
