"""阶梯：容差 τ 逐级放松，每一级把**全部搜索器**重跑一遍。

## 为什么搜索要按 τ / 策略重跑，而不是跑一遍再筛

因为**搜索器看到的分数依赖 τ 与策略**（`objective.score_of`）：

- `reject`：不可行 ⇒ `-1.0`。τ 变了，谁可行就变了，贪心的下一步就变了；
- `penalty`：不可行 ⇒ 扣分，扣多少取决于 `drift − τ`。

若只跑一遍再把不可行的结果滤掉，那么 `penalty` 那一列会**与 `reject` 逐位相同** ——
「两种约束处理策略的差别」这条结论会变成「没有差别」，而原因是实现偷了懒。
⇒ **每一级、每种策略都是独立的一次搜索**；贵的第 1 层由 `EvalCache` 兜住，
所以 7×2=14 次重跑的边际成本接近 0（实测：第 1 次 1.7 s，之后每次 ~10 ms）。

## 参考点：**全空间唯一一个，整次运行不变**

超体积必须有参考点，而**各用各的参考点，超体积之间根本没法比**。

第一版按级取 `ref_k = min(该级可行域) − pad` —— 它满足了「同一级内所有搜索器共用」，
却违反了更要紧的那一条：**放宽容差时参考点会移动**，
于是阶梯上超体积的跳变**有一部分只是参考点动了**。
实测：τ=0 → 1e-16 时超体积从 0.0321 涨到 0.0540（+68%），
其中有多少来自「前沿变宽」、多少来自「参考点下移」，第一版分不开。

⇒ 改成：`ref_k = min(整个搜索空间在该维上的值) − reference_pad`。
它**只由空间决定**，与容差、与策略、与搜索器都无关 —— 整次运行一个常数，
落在 `search.json` 里。于是阶梯上的超体积变化**只可能来自可行域**。

## 交付物是前沿，不是点

`best` / `best_score` 两个字段都留着，但**都不是交付物**：

- `best`：可行解里「三维下降比例均值」最大的那个 —— **这是一个投影**，
  只为了①检查阶梯单调性、②给报告一句人话。它是声明的，不是隐含的；
- `best_score`：搜索器评分最高的那个 —— `penalty` 策略下**可能不可行**，
  留着正是为了把「评分高」与「可行」的差量测出来（`best is not best_score`）。

真正的交付物是 `frontier`：一组互不支配的计划，各自带着自己的口径。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from .objective import (
    OBJECTIVES, is_feasible, reference_point, score_of, vector,
)
from .pareto import (
    Point, dominates, fast_non_dominated_sort, hypervolume,
    non_dominated_indices, spacing,
)
from .rewrites import LEGAL_NAMES
from .schema import Plan, Trial
from .search import (
    EvalCache, Evaluation, SearchContext, Visits, budget_for, make_searcher,
)

#: 固定权重（等权）。Tchebycheff 需要一组权重，搜索器需要**一个**标量。
#: 等权是**声明的口径**：不偏向任何一维。进化的 ParEGO 会在此基础上每代重抽，
#: 那是它自己的事 —— 固定权重只是「搜索器看到的那个数」的基准口径。
_EQUAL_WEIGHTS = tuple(1.0 / len(OBJECTIVES) for _ in OBJECTIVES)

#: Tchebycheff 的理想点：「每一维都下降 100%」= `(1, 1, 1)`。
#: 它是一个**声明的上界**，不是从数据里取的 —— 从数据里取会让理想点
#: 随搜索进度漂移，同一个候选在不同时刻得到不同分数。
IDEAL: Point = tuple(1.0 for _ in OBJECTIVES)


def mean_gain(values: Sequence[float]) -> float:
    """三维下降比例的**均值**。**这是一个投影，只用于排序与单调性检查。**

    结果层绝不允许标量化 —— 这里允许，是因为它出现在「挑一个代表点」
    与「比两级之间谁高」这两个地方，而这两处的结论都同时带着完整的前沿。
    """
    return sum(values) / len(values)


# --------------------------------------------------------------------- 数据结构
@dataclass
class SearcherCell:
    """一个 (τ, 策略, 搜索器) 格子。**不含耗时** —— 耗时进 `benchmark.json`。"""

    searcher: str
    tolerance: float
    strategy: str
    budget: int
    used: int
    hits: int
    shared: int
    n_proposed: int
    n_unique: int
    n_feasible: int
    n_frontier: int
    n_frontier_distinct: int
    hypervolume: float
    hv_gap_vs_oracle: float
    spacing: float
    best: Trial | None
    best_score: Trial | None
    frontier: list[Trial] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "searcher": self.searcher,
            "tolerance": self.tolerance,
            "strategy": self.strategy,
            "budget": self.budget,
            "used": self.used,
            "hits": self.hits,
            "shared": self.shared,
            "n_proposed": self.n_proposed,
            "n_unique": self.n_unique,
            "n_feasible": self.n_feasible,
            "n_frontier": self.n_frontier,
            "n_frontier_distinct": self.n_frontier_distinct,
            "hypervolume": round(self.hypervolume, 12),
            "hv_gap_vs_oracle": (None if self.hv_gap_vs_oracle != self.hv_gap_vs_oracle
                                 else round(self.hv_gap_vs_oracle, 12)),
            "spacing": round(self.spacing, 12),
            "best": None if self.best is None else self.best.to_dict(),
            "best_score": None if self.best_score is None else self.best_score.to_dict(),
            "frontier": [t.to_dict() for t in self.frontier],
        }


@dataclass
class Rung:
    """阶梯的一级（= 一个容差）× 一种策略。"""

    tolerance: float
    strategy: str
    reference: Point
    n_feasible_space: int          # 全空间里可行的计划数（oracle 数的）
    cells: dict[str, SearcherCell] = field(default_factory=dict)

    def oracle(self) -> SearcherCell:
        return self.cells["oracle"]

    def best_mean_gain(self, searcher: str = "oracle") -> float:
        c = self.cells[searcher]
        return float("nan") if c.best is None else mean_gain(c.best.objectives)

    def to_dict(self) -> dict[str, Any]:
        gains: dict[str, float] = {}
        for k in sorted(self.cells):
            v = self.cells[k]
            if v.best is not None:
                gains[k] = round(mean_gain(v.best.objectives), 12)
        return {
            "tolerance": self.tolerance,
            "strategy": self.strategy,
            "reference": [round(x, 12) for x in self.reference],
            "n_feasible_space": self.n_feasible_space,
            "best_mean_gain": gains,
            "cells": {k: self.cells[k].to_dict() for k in sorted(self.cells)},
        }


# --------------------------------------------------------------------- 单格搜索
def _make_scorers(cfg: dict[str, Any], tolerance: float, strategy: str):
    c = cfg["constraints"]
    ds = float(c["drift_scale"])
    pw = float(c["penalty_weight"])

    def _one(res, weights) -> float:
        ok = is_feasible(res.drift, tolerance)
        return score_of(ok, res.values, strategy=strategy, drift=res.drift,
                        tolerance=tolerance, drift_scale=ds,
                        penalty_weight=pw, weights=weights, ideal=IDEAL)

    def score(res) -> Evaluation:
        ok = is_feasible(res.drift, tolerance)
        return Evaluation(res, ok, _one(res, _EQUAL_WEIGHTS))

    def rescore(res, weights) -> float:
        return _one(res, weights)

    return score, rescore


def _dedupe(evs: Sequence[Evaluation]) -> list[Evaluation]:
    """按 `plan_id` 去重，**保持首次出现的顺序**。

    顺序不能靠 `set` —— `set` 的迭代顺序会让「同一个搜索器跑两次」给出
    不同的 `best`（当两个候选分数完全相同时，`max` 取先出现的那个）。
    """
    seen: set[str] = set()
    out: list[Evaluation] = []
    for e in evs:
        pid = e.plan.id
        if pid in seen:
            continue
        seen.add(pid)
        out.append(e)
    return out


def _to_trial(ev: Evaluation, searcher: str, index: int) -> Trial:
    m = ev.result.metrics
    return Trial(
        plan=ev.plan, searcher=searcher, index=index,
        feasible=ev.feasible, drift=ev.result.drift,
        objectives=tuple(ev.values), score=ev.score,
        applied=ev.result.applied,
        nodes=int(m.get("nodes", 0)), launches=int(m.get("launches", 0)),
        latency_us=float(m.get("latency_us", 0.0)),
        peak_bytes=int(m.get("peak_bytes", 0)),
        reject_reason=ev.result.reason,
    )


def run_rung(cfg: dict[str, Any], cache: EvalCache, tolerance: float,
             strategy: str, searcher_names: Sequence[str],
             timings: dict[str, float],
             reference: Point | None = None) -> Rung:
    """跑一级。**oracle 放在最后** —— 它是参照物，让它先去填缓存
    会让另外四个搜索器的 `shared` 全部变成「搭了 oracle 的车」，
    那样就看不出「四个搜索器之间互相省了多少」。

    `reference` 必须由调用方传入**全空间唯一的那一个**（见模块文档）。
    不传时会退化成按级取 —— 那是一个**已知有偏**的降级，只在单独调试时用。
    """
    score, rescore = _make_scorers(cfg, tolerance, strategy)
    ctx = SearchContext(LEGAL_NAMES, int(cfg["search"]["max_depth"]), cache,
                        score, rescore)

    order = [n for n in searcher_names if n != "oracle"]
    if "oracle" in searcher_names:
        order.append("oracle")

    cells: dict[str, SearcherCell] = {}
    raw: dict[str, list[Evaluation]] = {}
    for name in order:
        s = make_searcher(name, seed=cfg["seed"], params=cfg["search"])
        budget = budget_for(name, ctx, int(cfg["search"]["budget_per_searcher"]))
        v = Visits(ctx, budget)
        t0 = time.perf_counter()
        out = s.search(ctx, v)
        timings[f"{tolerance}|{strategy}|{name}"] = time.perf_counter() - t0
        raw[name] = out
        cells[name] = SearcherCell(
            searcher=name, tolerance=tolerance, strategy=strategy,
            budget=v.budget, used=v.used, hits=v.hits, shared=v.shared,
            n_proposed=len(out), n_unique=0, n_feasible=0, n_frontier=0,
            n_frontier_distinct=0,
            hypervolume=0.0, hv_gap_vs_oracle=float("nan"), spacing=0.0,
            best=None, best_score=None,
        )

    orc = raw.get("oracle")
    if orc is None:
        raise ValueError("run_rung 需要 oracle 作为参照物")
    feas_all = [e.values for e in orc if e.feasible]
    pad = float(cfg["search"]["reference_pad"])
    if reference is not None:
        ref = tuple(reference)
    elif feas_all:
        ref = tuple(min(p[k] for p in feas_all) - abs(pad)
                    for k in range(len(OBJECTIVES)))
    else:
        ref = reference_point(pad)

    # ---- 逐搜索器汇总
    for name in order:
        # ⚠️ **必须按 plan_id 去重**。搜索器会把「重复提案」的读数也收进返回值
        #    （`Visits` 对重复提案返回同一个 `Evaluation`），而重复提案不是新候选。
        #    实测抓过：`evolution` 报出 `n_feasible = 171`，
        #    而 τ=0 的可行域只有 **85** 个计划 —— **一个自相矛盾的数字**。
        #    它不报错、不影响前沿与超体积（重复点非支配彼此，体积不变），
        #    只让「评估了多少候选」这一列变成假的。
        out = _dedupe(raw[name])
        feas = [e for e in out if e.feasible]
        pts = [e.values for e in feas]
        idx = non_dominated_indices(pts) if pts else []
        front = [feas[i] for i in idx]
        fpts = [e.values for e in front]
        hv = hypervolume(fpts, ref) if fpts else 0.0

        # 按可行性挑「交出去的答案」；按评分挑「搜索器自己以为的答案」。
        best = max(feas, key=lambda e: (mean_gain(e.values),
                                        tuple(-x for x in e.values))) if feas else None
        best_score = max(out, key=lambda e: (e.score, e.plan.rules)) if out else None

        c = cells[name]
        c.n_proposed = len(raw[name])          # 原始提案数（含重复）
        c.n_unique = len(out)
        c.n_feasible = len(feas)
        c.n_frontier = len(front)
        # **前沿的「有效大小」**：不同的目标向量个数。
        # 实测：12 个前沿计划只有 **2** 个不同的权衡点 —— 其余 10 个是
        # 「换个顺序得到同一张图」。按计划重数报前沿大小，会被顺序冗余灌水，
        # 而 `min_frontier_points` 那条门槛会**靠重复点通过**。
        c.n_frontier_distinct = len({tuple(round(x, 12) for x in p) for p in fpts})
        c.hypervolume = hv
        c.spacing = spacing(fpts)
        c.best = None if best is None else _to_trial(best, name, out.index(best))
        c.best_score = (None if best_score is None
                        else _to_trial(best_score, name, out.index(best_score)))
        c.frontier = [_to_trial(e, name, i) for i, e in enumerate(front)]

    # ---- 支配关系（前沿成员各自被多少条前沿成员支配 ⇒ 恒为 0，这是保险丝）
    for name in order:
        c = cells[name]
        pts = [t.objectives for t in c.frontier]
        for i, t in enumerate(c.frontier):
            t.dominated_by = sum(1 for j, q in enumerate(pts)
                                 if j != i and dominates(q, pts[i]))
        layers = fast_non_dominated_sort(pts) if pts else []
        for li, layer in enumerate(layers):
            for i in layer:
                c.frontier[i].rank = li
        for t in c.frontier:
            t.on_frontier = True

    # ---- 相对 oracle 的超体积缺口
    hv_o = cells["oracle"].hypervolume
    for name in order:
        c = cells[name]
        c.hv_gap_vs_oracle = ((hv_o - c.hypervolume) / hv_o
                              if hv_o > 0 else float("nan"))

    return Rung(tolerance=tolerance, strategy=strategy, reference=ref,
                n_feasible_space=len(feas_all), cells=cells)


def space_reference(cfg: dict[str, Any], cache: EvalCache,
                    plan_rules: Sequence[Sequence[str]]) -> Point:
    """**全空间唯一的参考点**：`min(整个空间在该维上的值) − reference_pad`。

    它只由「空间 + 成本模型」决定，与容差 / 策略 / 搜索器全部无关。
    于是阶梯上超体积的任何变化**只可能来自可行域** ——
    这是让「放宽容差买到多少前沿宽度」这个问题**可回答**的前提。
    """
    pad = abs(float(cfg["search"]["reference_pad"]))
    res = [cache.get(Plan(tuple(r))) for r in plan_rules]
    return tuple(min(r.values[k] for r in res) - pad for k in range(len(OBJECTIVES)))


def run_ladder(cfg: dict[str, Any], cache: EvalCache, reference: Point
               ) -> tuple[list[Rung], dict[str, float]]:
    """整条阶梯：`ladder.tolerances` × `constraints.strategy` 的每一格。"""
    names = list(cfg["search"]["searchers"]) + ["oracle"]
    timings: dict[str, float] = {}
    rungs: list[Rung] = []
    for strategy in ("reject", "penalty"):
        for tol in [float(x) for x in cfg["ladder"]["tolerances"]]:
            rungs.append(run_rung(cfg, cache, tol, strategy, names, timings,
                                  reference=reference))
    return rungs, timings


def ladder_monotonicity(rungs: Sequence[Rung], strategy: str) -> dict[str, Any]:
    """阶梯上两个读数：**最好均值**（高度）与**超体积**（宽度）。

    `best_mean_gain` 单调不减是**数学必然**（τ 放松 ⇒ 可行域扩张 ⇒ 上确界不减），
    所以它红了一定是实现错了 —— 这正是它值得当门槛的原因。

    `active_steps` 却**必须算在超体积上**，不能算在最好均值上。实测：
    本项目的 10 级阶梯里，最好均值**一步都没动**（最优解本来就在 τ=0 的可行域里），
    而超体积有真实的台阶。原因很清楚 ——

    > **放宽容差买到的是前沿的宽度，不是最优点的高度。**

    用均值投影看，阶梯是一条平线；用超体积看，它才是阶梯。
    这两个读数必须都报，否则「阶梯测到了什么」取决于你挑了哪个投影。
    """
    rows = [r for r in rungs if r.strategy == strategy]
    rows = sorted(rows, key=lambda r: r.tolerance)
    vals = [r.best_mean_gain("oracle") for r in rows]
    hvs = [r.cells["oracle"].hypervolume for r in rows]

    viol: list[dict[str, Any]] = []
    active = 0
    for i in range(1, len(vals)):
        if vals[i] < vals[i - 1] - 1e-12:
            viol.append({"from": rows[i - 1].tolerance, "to": rows[i].tolerance,
                         "drop": vals[i - 1] - vals[i]})
    for i in range(1, len(hvs)):
        if hvs[i] > hvs[i - 1] + 1e-15:
            active += 1

    hv_viol = [{"from": rows[i - 1].tolerance, "to": rows[i].tolerance,
                "drop": hvs[i - 1] - hvs[i]}
               for i in range(1, len(hvs)) if hvs[i] < hvs[i - 1] - 1e-15]

    feas = [r.n_feasible_space for r in rows]
    feas_steps = sum(1 for i in range(1, len(feas)) if feas[i] > feas[i - 1])

    return {
        "strategy": strategy,
        "tolerances": [r.tolerance for r in rows],
        "best_mean_gain": [round(v, 12) for v in vals],
        "hypervolume": [round(v, 12) for v in hvs],
        "n_feasible_space": feas,
        "n_frontier": [r.cells["oracle"].n_frontier for r in rows],
        "active_steps": active,
        "flat_steps": len(rows) - 1 - active,
        "feasible_steps": feas_steps,
        "violations": viol,
        "hypervolume_violations": hv_viol,
        "monotone": not viol and not hv_viol,
        "height_monotone": not viol,
        "width_monotone": not hv_viol,
    }


def run_budget_sweep(cfg: dict[str, Any], cache: EvalCache, reference: Point
                     ) -> dict[str, Any]:
    """预算扫描：`超体积(budget)` 逐搜索器。

    ## 为什么单一预算点不是比较，是轶事

    实测（本项目）：`random` 在 40~60 次评估就把 oracle 的超体积拿满了。
    于是主预算 120 下的「谁更强」实际上在比**运气**，而 `beam` 因为
    「展开到一半预算就用完」反而显得最差 —— 那个排名是预算截断位置的产物，
    不是策略能力的读数。

    扫描把这件事变成一条曲线：**策略的贡献与预算的贡献是耦合的**，
    最优点会随预算换人（实测：10 次时 evolution 最好、20 次时 greedy 最好、
    40 次起 random 追上）。这个结论只能从曲线上读出来。

    主预算与紧预算都由 `config` 声明，两者都要报 ——
    **只报一个预算点，等于替读者选了「哪个预算下比较」这个前提。**
    """
    budgets = [int(b) for b in cfg["budget_sweep"]["budgets"]]
    strategy = str(cfg["budget_sweep"]["strategy"])
    tolerance = float(cfg["budget_sweep"]["tolerance"])
    names = list(cfg["search"]["searchers"]) + ["oracle"]
    timings: dict[str, float] = {}

    rows: list[dict[str, Any]] = []
    for b in budgets:
        sub = dict(cfg)
        sub["search"] = dict(cfg["search"])
        sub["search"]["budget_per_searcher"] = b
        rung = run_rung(sub, cache, tolerance, strategy, names, timings,
                        reference=reference)
        cells = {n: rung.cells[n] for n in names}
        hv_o = cells["oracle"].hypervolume
        rows.append({
            "budget": b,
            "n_feasible_space": rung.n_feasible_space,
            "reference": [round(x, 12) for x in rung.reference],
            "oracle_hypervolume": round(hv_o, 12),
            "oracle_n_frontier": cells["oracle"].n_frontier,
            "cells": {
                n: {
                    "used": cells[n].used,
                    "n_frontier": cells[n].n_frontier,
                    "hypervolume": round(cells[n].hypervolume, 12),
                    "hv_gap_vs_oracle": (None if cells[n].hv_gap_vs_oracle !=
                                         cells[n].hv_gap_vs_oracle
                                         else round(cells[n].hv_gap_vs_oracle, 12)),
                    "hv_per_eval": (round(cells[n].hypervolume / cells[n].used, 15)
                                    if cells[n].used else None),
                    "best_mean_gain": (round(mean_gain(cells[n].best.objectives), 12)
                                       if cells[n].best else None),
                }
                for n in names
            },
        })

    # 饱和点：random 第一次达到 oracle 超体积 99.9% 的预算
    sat = None
    for r in rows:
        c = r["cells"]["random"]
        if (r["oracle_hypervolume"] > 0
                and c["hypervolume"] >= 0.999 * r["oracle_hypervolume"]):
            sat = r["budget"]
            break

    # 每个预算下「最好的非随机搜索器」以及它相对 random 的增益
    margins: list[dict[str, Any]] = []
    for r in rows:
        base = r["cells"]["random"]["hypervolume"]
        others = {n: c["hypervolume"] for n, c in r["cells"].items()
                  if n not in ("random", "oracle")}
        best = max(others, key=lambda n: others[n]) if others else None
        gain = ((others[best] - base) / base) if (best and base > 0) else 0.0
        margins.append({"budget": r["budget"], "best_searcher": best,
                        "random_hypervolume": round(base, 12),
                        "best_hypervolume": round(others[best], 12) if best else None,
                        "gain_over_random": round(gain, 12)})

    return {
        "budgets": budgets,
        "strategy": strategy,
        "tolerance": tolerance,
        "searchers": names,
        "rows": rows,
        "margins": margins,
        "random_saturation_budget": sat,
        "tight_budget": int(cfg["search"]["tight_budget"]),
        "timings": timings,
    }
    rows = [r for r in rungs if r.strategy == strategy]
    rows = sorted(rows, key=lambda r: r.tolerance)
    vals = [r.best_mean_gain("oracle") for r in rows]
    viol: list[dict[str, Any]] = []
    for i in range(1, len(vals)):
        if vals[i] < vals[i - 1] - 1e-12:
            viol.append({"from": rows[i - 1].tolerance, "to": rows[i].tolerance,
                         "drop": vals[i - 1] - vals[i]})
    return {
        "strategy": strategy,
        "tolerances": [r.tolerance for r in rows],
        "best_mean_gain": [round(v, 12) for v in vals],
        "violations": viol,
        "monotone": not viol,
    }
