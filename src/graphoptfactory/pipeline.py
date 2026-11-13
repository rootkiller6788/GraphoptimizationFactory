"""编排：把一次 `run` 的全部产物组装出来。

## 一条纪律：渲染器只吃产物字典

`run()` 返回的 `Run.products` 是**纯数据**（可 JSON 序列化）。
`report.render_markdown(products)` / `render_html(products)` **只读这个字典**，
不读 `baseline` / `cache` / `evaluator` 这些运行期对象。

买到两样：

1. `visualize` 能从 `out/` 重建报告，**一行都不重跑流水线**；
2. 它证明的东西更强了 —— 从「流水线是确定的」变成「**产物足以重造报告**」。
   对抗用例：把一次 run 折成产物 → 从产物重建 → 再渲染，与直接渲染**逐位比**。
   （`L5-quantfactory` 靠这条抓出过 `render_html` 里 `len(run.products) - 2`
   在产物还没填满时印出「其余 −2 个文件」。**负数比报错更危险，因为它看起来像个数字。**）

## 什么进版本摘要，什么不进

`VERSIONED_FILES` 显式列出来。排除三类：

- `version.json`：自指 + 时间戳；
- `benchmark.json`：**耗时**（随机器变，算进去版本号就失去意义）；
- `report.md` / `report.html`：它们是**视图**，算进去会让版本号依赖排版细节。

⇒ 于是「两次 `run` 只有 `version.json` 不同」这条核对要**精确表述**为
「两次 `run` 只有 `version.json`（时间戳）与 `benchmark.json`（耗时）不同，
其余 N 个结论性产物逐位相同」。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from .adversarial import run_adversarial
from .benchmark import cost_sweep, wallclock_probe
from .cost import COST_SOURCE, metrics
from .evaluate import PlanEvaluator
from .graph import build_baseline, build_probes, shape_table
from .ladder import (
    Rung, ladder_monotonicity, mean_gain, run_budget_sweep, run_ladder,
    space_reference,
)
from .lineage import lineage_graph, order_sensitivity, rule_table
from .objective import OBJECTIVES
from .pareto import correlation_matrix
from .rewrites import CONTROL_NAMES, LEGAL_NAMES
from .schema import Plan
from .search import EvalCache, declared_params
from .semantics import run_outputs

#: 进版本摘要的产物（**顺序固定**，摘要按这个顺序拼）。
VERSIONED_FILES: tuple[str, ...] = (
    "config.json",
    "baseline.json",
    "space.json",
    "lineage.json",
    "adversarial.json",
    "search.json",
    "budget_sweep.json",
    "ladder.json",
    "expectations.json",
    "acceptance.json",
    "summary.json",
)

#: 落盘但不进摘要（耗时 / 视图 / 自指）。
UNVERSIONED_FILES: tuple[str, ...] = (
    "version.json", "benchmark.json", "report.md", "report.html",
)


@dataclass
class Run:
    cfg: dict[str, Any]
    products: dict[str, Any]
    timings: dict[str, float] = field(default_factory=dict)
    rungs: list[Rung] = field(default_factory=list)
    cache: EvalCache | None = None


def _round_tensors(ts: Sequence[Any]) -> list[Any]:
    """`tuple[Tensor, ...]` → 可 JSON 化的嵌套列表。**逐层降维，别少写一层。**"""
    return [[[[round(v, 12) for v in row] for row in t] for t in probe]
            for probe in ts]


def _baseline_product(baseline, cfg, base_m, base_out) -> dict[str, Any]:
    shapes = shape_table(baseline, cfg)
    op_counts: dict[str, int] = {}
    for n in baseline.nodes.values():
        op_counts[n.op] = op_counts.get(n.op, 0) + 1
    return {
        "digest": baseline.digest(),
        "n_nodes": len(baseline.nodes),
        "op_counts": {k: op_counts[k] for k in sorted(op_counts)},
        "n_outputs": len(baseline.outputs),
        "shape_table": {k: list(v) for k, v in sorted(shapes.items())},
        "metrics": {k: base_m[k] for k in sorted(base_m)},
        "cost_source": {k: COST_SOURCE[k] for k in sorted(COST_SOURCE)},
        "outputs": _round_tensors(base_out),
        "graph": baseline.to_dict(),
    }


def _space_product(cache: EvalCache, baseline, cfg, evaluator: PlanEvaluator,
                   rung0: Rung, plan_rules: Sequence[Sequence[str]],
                   reference: Any) -> dict[str, Any]:
    """搜索空间本身的刻画：多大、有多少不同的图、目标之间冗不冗余。"""
    plans = [Plan(tuple(x)) for x in plan_rules]
    res = [cache.get(p) for p in plans]
    digests = {r.digest for r in res if r.digest}
    vecs = {tuple(round(x, 12) for x in r.values) for r in res}

    # 目标相关矩阵：在 **τ=0 的可行集**上算（不可行候选不参与结论）
    feas = [r.values for r in res if r.drift == 0.0]
    corr = correlation_matrix(feas, OBJECTIVES) if len(feas) >= 3 else {}

    # 各深度的最好（按均值）
    by_depth: dict[str, dict[str, Any]] = {}
    for k in range(1, int(cfg["search"]["max_depth"]) + 1):
        sub = [r for r in res if r.plan.depth == k]
        if not sub:
            continue
        best = max(sub, key=lambda r: mean_gain(r.values))
        by_depth[str(k)] = {
            "n_plans": len(sub),
            "best_mean_gain": round(mean_gain(best.values), 12),
            "best_rules": list(best.plan.rules),
            "best_objectives": [round(x, 12) for x in best.values],
        }

    return {
        "space_size": len(plans),
        "n_plans_evaluated": len(cache.store),
        "n_distinct_digests": len(digests),
        "n_distinct_objective_vectors": len(vecs),
        "n_feasible_at_tau0": len(feas),
        "n_feasible_at_tau0_distinct": len({tuple(round(x, 12) for x in v)
                                            for v in feas}),
        "objective_correlation": corr,
        "depth_best": by_depth,
        "evaluator": evaluator.diagnostics(),
        "n_frontier_at_tau0": rung0.cells["oracle"].n_frontier,
        "hypervolume_at_tau0": round(rung0.cells["oracle"].hypervolume, 12),
        "reference": [round(x, 12) for x in reference],
        "reference_note": "全空间唯一参考点：min(整个空间) − reference_pad。"
                          "与容差 / 策略 / 搜索器全部无关。",
    }


def _all_rules(cfg: dict[str, Any]) -> list[tuple[str, ...]]:
    out: list[tuple[str, ...]] = []
    max_depth = int(cfg["search"]["max_depth"])
    rules = tuple(LEGAL_NAMES)

    def rec(cur: tuple[str, ...]) -> None:
        if cur:
            out.append(cur)
        if len(cur) >= max_depth:
            return
        for r in rules:
            if r not in cur:
                rec(cur + (r,))

    rec(())
    return out


def _search_product(cfg: dict[str, Any], cache: EvalCache,
                    space_size: int) -> dict[str, Any]:
    """搜索器清单与**预算不对称**（必须显式，见 `search/oracle.py`）。"""
    names = list(cfg["search"]["searchers"]) + ["oracle"]
    budget = int(cfg["search"]["budget_per_searcher"])
    return {
        "searchers": names,
        "params_declared": {k: list(v) for k, v in sorted(declared_params().items())},
        "space_size": space_size,
        "budget_per_searcher": budget,
        "budget_asymmetry": {
            "oracle": space_size,
            "others": budget,
            "note": "oracle 的预算 = 空间大小，其余 = budget_per_searcher。"
                    "这不是特权，是定义（否则 oracle 就不是 oracle，"
                    "而 oracle_dominates_all_searchers 会从必然变成可能）。",
        },
        "cache": {"unique_evaluations": cache.unique, "cache_hits": cache.hits},
        "constraints": dict(cfg["constraints"]),
        "semantics": dict(cfg["semantics"]),
        "scoring": {
            "kind": "equal_weight_tchebycheff",
            "weights": [1.0 / len(OBJECTIVES)] * len(OBJECTIVES),
            "ideal": [1.0] * len(OBJECTIVES),
            "note": "固定权重只决定「搜索器看到的那个数」；交付物是未加权的三目标前沿。",
        },
    }


def _frontier_product(rungs: Sequence[Rung], cfg: dict[str, Any]) -> dict[str, Any]:
    """交付物本体：每一级、每种策略下 oracle 的前沿（计划 + 口径）。"""
    out: dict[str, Any] = {}
    for r in rungs:
        key = f"{r.strategy}@{r.tolerance:g}"
        out[key] = {
            "tolerance": r.tolerance,
            "strategy": r.strategy,
            "reference": [round(x, 12) for x in r.reference],
            "n_feasible_space": r.n_feasible_space,
            "hypervolume": round(r.oracle().hypervolume, 12),
            "points": [
                {
                    "plan_id": t.plan_id,
                    "rules": list(t.plan.rules),
                    "drift": t.drift,
                    "objectives": [round(x, 12) for x in t.objectives],
                    "mean_gain": round(mean_gain(t.objectives), 12),
                    "nodes": t.nodes,
                    "launches": t.launches,
                    "latency_us": round(t.latency_us, 9),
                    "peak_bytes": t.peak_bytes,
                }
                for t in r.oracle().frontier
            ],
        }
    return out


def _best_answer_product(rungs: Sequence[Rung]) -> dict[str, Any]:
    """τ=0 / reject 下 oracle 交出去的那个答案（报告的头条数字来自这里）。"""
    r0 = [r for r in rungs if r.strategy == "reject" and r.tolerance == 0.0]
    if not r0:
        return {}
    c = r0[0].oracle()
    if c.best is None:
        return {}
    return {
        "plan_id": c.best.plan_id,
        "rules": list(c.best.plan.rules),
        "drift": c.best.drift,
        "objectives": [round(x, 12) for x in c.best.objectives],
        "mean_gain": round(mean_gain(c.best.objectives), 12),
        "nodes": c.best.nodes,
        "launches": c.best.launches,
        "latency_us": round(c.best.latency_us, 9),
        "peak_bytes": c.best.peak_bytes,
    }


def build(cfg: dict[str, Any]) -> Run:
    """跑完一次完整流水线，返回全部产物。**不写盘**（写盘在 `emit.py`）。"""
    t_start = time.perf_counter()
    timings: dict[str, float] = {}

    baseline = build_baseline(cfg)
    probes = build_probes(cfg)
    shapes = shape_table(baseline, cfg)
    base_m = metrics(baseline, shapes, cfg)
    base_out = [run_outputs(baseline, f) for f in probes]

    evaluator = PlanEvaluator(cfg, baseline, probes, base_m)
    cache = EvalCache(evaluator)

    tolerances = [float(x) for x in cfg["ladder"]["tolerances"]]

    t0 = time.perf_counter()
    rules = rule_table(cfg, baseline, probes, tolerances)
    lin = lineage_graph(cache, baseline, cfg)
    order = order_sensitivity(cache, cfg)
    timings["lineage"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    adv = run_adversarial(cfg, baseline, probes, tolerances)
    timings["adversarial"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    # 参考点必须**先**算出来：它只由空间决定，且必须整次运行唯一
    # （按级取会让「放宽容差」同时移动参考点，超体积的变化就分不清来源）。
    all_plans = _all_rules(cfg)
    reference = space_reference(cfg, cache, all_plans)
    rungs, rung_timings = run_ladder(cfg, cache, reference)
    timings["ladder"] = time.perf_counter() - t0
    timings.update({f"rung:{k}": v for k, v in rung_timings.items()})

    rung0 = [r for r in rungs if r.strategy == "reject" and r.tolerance == 0.0][0]
    space = _space_product(cache, baseline, cfg, evaluator, rung0, all_plans,
                           reference)

    t0 = time.perf_counter()
    sweep = run_budget_sweep(cfg, cache, reference)
    timings["budget_sweep"] = time.perf_counter() - t0
    timings.update({f"sweep:{k}": v for k, v in sweep["timings"].items()})

    # ---- 基准：解析成本 vs 实测耗时（**只对前沿成员测**，省时间且信息量足）
    probe_plans = [Plan(tuple(p["rules"])) for p in _frontier_product(rungs, cfg)
                   .get("reject@0", {}).get("points", [])]
    probe_plans = probe_plans[:8]
    t0 = time.perf_counter()
    bench = {"wallclock": wallclock_probe(cfg, baseline, probes, probe_plans)}
    best_metrics = _best_metrics_for_sweep(rungs, cache)
    if best_metrics is not None:
        bench["cost_sweep"] = cost_sweep(cfg, base_m, best_metrics)
    timings["benchmark"] = time.perf_counter() - t0
    bench["timings"] = dict(timings)
    timings["total"] = time.perf_counter() - t_start
    bench["timings"] = dict(timings)

    products: dict[str, Any] = {
        "config.json": {"resolved": cfg, "protocol_digest": None},
        "baseline.json": _baseline_product(baseline, cfg, base_m, base_out),
        "space.json": space,
        "lineage.json": {
            "rules": rules,
            "legal_names": list(LEGAL_NAMES),
            "control_names": list(CONTROL_NAMES),
            "graph": lin,
            "order_sensitivity": order,
        },
        "adversarial.json": adv,
        "search.json": _search_product(cfg, cache, space["space_size"]),
        "budget_sweep.json": {k: v for k, v in sweep.items() if k != "timings"},
        "ladder.json": {
            "rungs": [r.to_dict() for r in rungs],
            "monotonicity": [ladder_monotonicity(rungs, s)
                             for s in ("reject", "penalty")],
            "frontier": _frontier_product(rungs, cfg),
            "best_answer": _best_answer_product(rungs),
        },
        "benchmark.json": bench,
        "summary.json": {},      # 下面填（依赖 expectations / acceptance）
    }

    # 协议指纹：**只覆盖影响结论的字段**，不含路径
    from .config import protocol_digest
    products["config.json"]["protocol_digest"] = protocol_digest(cfg)
    bench["timings"] = dict(timings)

    return Run(cfg=cfg, products=products, timings=timings, rungs=rungs, cache=cache)


def _best_metrics_for_sweep(rungs: Sequence[Rung],
                            cache: EvalCache) -> dict[str, Any] | None:
    r0 = [r for r in rungs if r.strategy == "reject" and r.tolerance == 0.0]
    if not r0 or r0[0].oracle().best is None:
        return None
    return cache.get(Plan(tuple(r0[0].oracle().best.plan.rules))).metrics


def finish(run: Run) -> Run:
    """补上依赖 expectations / acceptance 的产物，并算 summary。

    **分两步是为了让 `summary` 能引用验收结果** —— 而 `summary` 又进版本摘要，
    所以「验收通过与否」会进版本号。这是刻意的：
    两个版本号相同 ⇒ 连验收结论都相同，而不只是数字相同。
    """
    from .acceptance import run_acceptance
    from .expectations import run_expectations
    from .invariants import registered_cases, run_cases

    products = run.products
    exp_all = run_expectations(products)
    # ⚠️ **时序类期望必须落进 benchmark.json**（不进版本摘要的产物）。
    #    实测抓过：`analytic_latency_tracks_wallclock` 的证据里嵌着每次求值的
    #    微秒数，而 `expectations.json` 是**进摘要**的 ⇒
    #    「同一份输入跑两次」得到了两个版本号，版本号的意义当场失效。
    products["benchmark.json"]["timing_expectations"] = exp_all["timing_items"]
    products["expectations.json"] = {k: v for k, v in exp_all.items()
                                     if k != "timing_items"}
    products["acceptance.json"] = run_acceptance(products)

    exp = products["expectations.json"]
    acc = products["acceptance.json"]["checks"]
    ladder = products["ladder.json"]
    space = products["space.json"]
    adv = products["adversarial.json"]

    best = ladder["best_answer"]
    r0 = ladder["frontier"].get("reject@0", {})
    ref_metrics = products["baseline.json"]["metrics"]
    products["summary.json"] = {
        "protocol_digest": products["config.json"]["protocol_digest"],
        "space_size": space["space_size"],
        "n_distinct_digests": space["n_distinct_digests"],
        "baseline": {
            "nodes": ref_metrics["nodes"],
            "launches": ref_metrics["launches"],
            "latency_us": round(ref_metrics["latency_us"], 9),
            "peak_bytes": ref_metrics["peak_bytes"],
        },
        "best_answer": best,
        "frontier_points_at_tau0": len(r0.get("points", [])),
        "hypervolume_at_tau0": r0.get("hypervolume"),
        "safe_window": exp["derived"]["safe_window"],
        "illegal_accepted_at_default_tolerance":
            adv["n_accepted_at_default_tolerance"],
        "n_legal_bit_identical": sum(1 for r in products["lineage.json"]["rules"]
                                     if r["kind"] == "legal" and r["bit_identical"]),
        "n_legal_inexact": sum(1 for r in products["lineage.json"]["rules"]
                               if r["kind"] == "legal" and not r["bit_identical"]),
        "order_sensitive_unordered_pairs":
            products["lineage.json"]["order_sensitivity"]["n_order_sensitive_unordered"],
        "budget_sweep": {
            "budgets": products["budget_sweep.json"]["budgets"],
            "random_saturation_budget":
                products["budget_sweep.json"]["random_saturation_budget"],
            "tight_budget": products["budget_sweep.json"]["tight_budget"],
            "tight_budget_best_gain_over_random":
                next((m["gain_over_random"] for m in
                      products["budget_sweep.json"]["margins"]
                      if m["budget"] == products["budget_sweep.json"]["tight_budget"]),
                     None),
        },
        "ladder_active_steps": {
            r["strategy"]: r["active_steps"]
            for r in products["ladder.json"]["monotonicity"]
        },
        "ladder_feasible_steps": {
            r["strategy"]: r["feasible_steps"]
            for r in products["ladder.json"]["monotonicity"]
        },
        "expectations": {
            "n": exp["n"],
            "n_held": exp["n_held"],
            "n_refuted": exp["n_refuted"],
            "n_unmeasurable": exp["n_unmeasurable"],
            #: 时序类期望的**条数**是确定性的（内容不是）——
            #: 它们的读数在 `benchmark.json` 里。
            "n_timing_dependent": exp["n_timing"],
        },
        "acceptance": {
            "n": len(acc),
            "n_passed": sum(1 for c in acc if c["passed"]),
            "n_failed": sum(1 for c in acc if not c["passed"]),
            #: 对抗用例的条数也进摘要 —— 但它**不是**门槛，所以不进 `n`。
            #: 口径：门槛是达标线（可调），对抗用例是不变量断言（不可调）。
            #: ⚠️ 这里读**注册表**而不是 `acceptance.json`：摘要必须排在
            #:    用例之前算（用例要读摘要），此刻 `cases` 还没被填进去 ——
            #:    读产物会永远读到 0，而 0 看起来像个合法的数。
            "n_cases": len(registered_cases()),
            "n_cases_failed": None,   # 用例跑完后回填（见函数末尾）
        },
        "cache": products["search.json"]["cache"],
    }

    # ---- 对抗用例：**必须放在 summary 之后**
    # 它们要读 `summary.json`（「摘要是不是别的产物的投影」本身就是一条用例），
    # 而 `summary.json` 又要读验收结论 ⇒ 顺序只能是「门槛 → 摘要 → 用例」。
    # 反过来的话那条用例只能跳过，而「跳过」与「通过」在输出里长得一模一样。
    cases = run_cases(products)
    products["acceptance.json"].update(cases)
    # 回填摘要里那个只有跑完才知道的数（条数已知，失败数未知）。
    products["summary.json"]["acceptance"]["n_cases_failed"] = cases["n_cases_failed"]
    return run
