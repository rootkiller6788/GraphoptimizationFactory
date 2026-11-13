"""基准：耗时读数，以及「解析成本模型能不能预测实际耗时」这一条。

## 这个文件里的东西**全部排除在版本摘要之外**

耗时随机器变。把耗时算进版本号，「同一份输入跑两次得到同一个版本号」
这条性质立刻失效 —— 而版本号的全部意义就是那句话。
（同 `L5-modeloptimizationfactory`：`benchmark.json` 不进 `VERSIONED_FILES`。）

## 解析延迟 vs 实测耗时：一条**预期会被推翻**的期望

`cost.metrics` 给出的 `latency_us` 是**解析的**：它由 `launch_overhead_us`、
`flops_per_us`、`bytes_per_us` 三个假设常数算出。而实测耗时在纯 Python 求值器上
由「Python 解释开销 + 张量尺寸」决定。两者同向吗？

这个问题值得单独测，因为它决定**成本模型能不能被当作代理目标**：

- 若秩相关高 ⇒ 用解析成本当搜索目标是有依据的；
- 若秩相关低 ⇒ 搜索优化的是一张**假设的表**，不是真实耗时 ——
  这个结论必须写进报告，否则「延迟下降 31.9%」会被读成「跑得快了 31.9%」。

⇒ 实测只做**重复取最小值**（不是均值）：最小值最接近「没有调度噪声」的那次，
而我们要问的是「这个图的固有耗时」，不是「这台机器此刻有多忙」。
"""

from __future__ import annotations

import time
from typing import Any, Sequence

from .cost import metrics
from .graph import shape_table
from .pareto import spearman
from .rewrites import apply_rule
from .schema import Graph, Plan, Tensor
from .semantics import run_outputs


def wallclock_probe(cfg: dict[str, Any], baseline: Graph, probes: Sequence[Tensor],
                    plans: Sequence[Plan], repeats: int = 3) -> dict[str, Any]:
    """对一组计划：解析延迟 vs 实测耗时（重复取最小）。"""
    rows: list[dict[str, Any]] = []
    for plan in plans:
        g = baseline
        for r in plan.rules:
            g, _ = apply_rule(g, r, cfg)
        shapes = shape_table(g, cfg)
        analytic = float(metrics(g, shapes, cfg)["latency_us"])

        best = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            for feed in probes:
                run_outputs(g, feed)
            best = min(best, time.perf_counter() - t0)
        rows.append({
            "plan_id": plan.id,
            "rules": list(plan.rules),
            "analytic_latency_us": analytic,
            "measured_us": best * 1e6,
        })

    a = [r["analytic_latency_us"] for r in rows]
    b = [r["measured_us"] for r in rows]
    rho = spearman(a, b)
    return {
        "n_plans": len(rows),
        "repeats": repeats,
        "rows": rows,
        "spearman_rho": rho,
        "rho_is_finite": rho == rho,
        "tracks": bool(rho == rho and rho >= 0.5),
    }


def cost_sweep(cfg: dict[str, Any], baseline: dict[str, Any],
               best: dict[str, Any]) -> list[dict[str, Any]]:
    """扫 `launch_overhead_us`：融合的收益有多少来自 launch 开销。

    一条**软假设**的系数扫描：凡「某机制买到 X」的结论，
    都要报告它在哪个系数区间才成立（同附录 C 244）。
    除 `launch_overhead_us` 外全部常数保持默认 —— 一次只动一个变量。
    """
    from .cost import sweep_launch_overhead
    return sweep_launch_overhead(
        cfg, [float(x) for x in cfg["cost_sweep"]["launch_overhead_us"]],
        baseline, best)
