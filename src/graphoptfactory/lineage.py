"""血缘：**重写图** —— 基线 → 每条规则 → 得到的图，逐条记账。

## 为什么血缘是本项目的一等产物

设计稿的核心命题是「L6 是总控 Optimizer，不负责写具体 Kernel」。
那么它交出去的东西必须**可追溯**：一条计划为什么更快，是因为哪几条重写、
每条各改了几个点、每条各自把漂移推到了多少。没有血缘，前沿上的一个点
就只是一个数对，接手的人无法判断它能不能信。

## 节点是**图指纹**，不是计划

`Graph.digest()` 只吃输出与常量，与节点 id 无关 —— 于是
「两条不同的计划得到同一张图」这件事**看得见**（CSE 与常量折叠在有些顺序下
会撞到同一张图）。用计划当节点的话，这种「殊途同归」会被记成两条独立路径。

## 计数口径：`hits` 是**真正改动**的次数

`common_subexpression` 第一版把「新建了一个节点」也算成命中，报出 49 次，
实际只合并了 2 个。**计数口径错了，读数就全错**，而且它不会报错 ——
只是让「这条规则改了多少地方」变成一个假数。
"""

from __future__ import annotations

from typing import Any, Sequence

from .cost import metrics
from .graph import shape_table
from .rewrites import CONTROL_RULES, LEGAL_NAMES, LEGAL_RULES, apply_rule
from .schema import Graph, GraphError, Plan, Tensor
from .semantics import validate
from .search import EvalCache


def rule_table(cfg: dict[str, Any], baseline: Graph, probes: Sequence[Tensor],
               tolerances: Sequence[float]) -> list[dict[str, Any]]:
    """逐规则（深度 1）：命中、漂移、逐位等价与否、在哪几级容差被接受。"""
    base_m = metrics(baseline, shape_table(baseline, cfg), cfg)
    rows: list[dict[str, Any]] = []
    for rule in LEGAL_RULES + CONTROL_RULES:
        g, hits = apply_rule(baseline, rule.name, cfg)
        try:
            m = metrics(g, shape_table(g, cfg), cfg)
        except GraphError as e:
            rows.append({
                "rule": rule.name, "kind": rule.kind, "note": rule.note,
                "hits": hits, "drift": float("inf"), "bit_identical": False,
                "digest": "", "error": str(e),
                "delta": {}, "accepted_at": {str(t): False for t in tolerances},
            })
            continue
        v = validate(baseline, g, probes, 0.0)
        rows.append({
            "rule": rule.name,
            "kind": rule.kind,
            "note": rule.note,
            "hits": hits,
            "drift": v.drift,
            "bit_identical": v.drift == 0.0,
            "digest": g.digest(),
            "error": v.reason if v.drift == float("inf") else "",
            "delta": {
                "nodes": m["nodes"] - base_m["nodes"],
                "launches": m["launches"] - base_m["launches"],
                "flops": m["flops"] - base_m["flops"],
                "bytes_moved": m["bytes_moved"] - base_m["bytes_moved"],
                "latency_us": round(m["latency_us"] - base_m["latency_us"], 9),
                "peak_bytes": m["peak_bytes"] - base_m["peak_bytes"],
            },
            "accepted_at": {str(t): bool(v.drift <= t) for t in tolerances},
        })
    return rows


def lineage_graph(cache: EvalCache, baseline: Graph, cfg: dict[str, Any]
                  ) -> dict[str, Any]:
    """重写图的节点与边。

    - 节点 = 图指纹（含基线）；
    - 边 = `(父图指纹) --规则--> (子图指纹)`，附命中数与漂移。

    边只从**空间内**的计划生成：深度 `k` 的计划，父节点是它的前 `k-1` 条规则
    所对应的图。所有前缀都在空间里（长度 1..k-1），所以不需要额外求值。
    """
    base_digest = baseline.digest()
    nodes: set[str] = {base_digest}
    edges: dict[tuple[str, str, str], dict[str, Any]] = {}
    n_plans = 0

    max_depth = int(cfg["search"]["max_depth"])
    rules = tuple(LEGAL_NAMES)

    def rec(cur: tuple[str, ...]) -> None:
        nonlocal n_plans
        if cur:
            n_plans += 1
            plan = Plan(cur)
            res = cache.get(plan)
            child = res.digest
            if not child:                       # 结构失败的候选没有指纹
                return
            nodes.add(child)
            parent_plan = Plan(cur[:-1])
            parent = (base_digest if not cur[:-1] else cache.get(parent_plan).digest)
            if not parent:
                return
            key = (parent, cur[-1], child)
            e = edges.get(key)
            if e is None:
                edges[key] = {
                    "from": parent, "rule": cur[-1], "to": child,
                    "hits": res.hits, "drift": res.drift, "count": 1,
                }
            else:
                e["count"] += 1
        if len(cur) >= max_depth:
            return
        for r in rules:
            if r not in cur:
                rec(cur + (r,))

    rec(())
    return {
        "baseline_digest": base_digest,
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "n_plans": n_plans,
        "nodes": sorted(nodes),
        "edges": [edges[k] for k in sorted(edges)],
    }


def order_sensitivity(cache: EvalCache, cfg: dict[str, Any]) -> dict[str, Any]:
    """规则顺序到底重不重要 —— **用全空间逐对比**，不是抽查。

    只比 2 元对（`P(8,2)=56` 个有序对 = 28 个无序对）：
    深度 3 的对比数量是 336×335，没有必要 —— 2 元对已经能回答
    「顺序是不是空间的一部分」这个问题。
    """
    rules = tuple(LEGAL_NAMES)
    differing: list[dict[str, Any]] = []
    total = 0
    for a in rules:
        for b in rules:
            if a == b:
                continue
            x = cache.get(Plan((a, b)))
            y = cache.get(Plan((b, a)))
            total += 1
            if x.values != y.values:
                differing.append({
                    "a_then_b": [a, b], "b_then_a": [b, a],
                    "values_ab": [round(v, 12) for v in x.values],
                    "values_ba": [round(v, 12) for v in y.values],
                    "max_abs_diff": round(max(abs(p - q) for p, q in
                                               zip(x.values, y.values)), 12),
                })
    return {
        "n_ordered_pairs": total,
        "n_unordered_pairs": total // 2,
        "n_order_sensitive_ordered": len(differing),
        "n_order_sensitive_unordered": len(differing) // 2,
        "examples": sorted(differing, key=lambda d: d["a_then_b"])[:8],
    }
