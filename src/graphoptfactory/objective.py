"""目标定义、归一化与**仅供搜索器内部**的标量化。

**标量化绝不允许出现在结果层。** 结果层一律用未加权的三目标向量；
一旦把三轴压成一个数，输出就不再是前沿，而是一个点 —— 直接违反设计稿的
「输出 Pareto 结果」。这条纪律写在这里，也写进 `report.py` 的模块文档。

三个目标都是**下降比例**（越大越好），口径统一、量纲可比：

| 目标 | 定义 | 含义 |
| --- | --- | --- |
| `speedup_gain` | `1 − cand_latency / base_latency` | 解析延迟下降比例 |
| `memory_saving` | `1 − cand_peak / base_peak` | 峰值内存下降比例 |
| `launch_reduction` | `1 − cand_launches / base_launches` | kernel launch 次数下降比例 |

三者都可能高度相关（融合同时降 launch 与访存）—— **这是要测的，不是要掩盖的**：
相关矩阵会原样落盘，冗余的那一维按「保留并写成一条被推翻的设计期望」处理。
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from .pareto import Point

OBJECTIVES: tuple[str, ...] = ("speedup_gain", "memory_saving", "launch_reduction")

OBJECTIVE_LABELS: dict[str, str] = {
    "speedup_gain": "延迟下降",
    "memory_saving": "内存下降",
    "launch_reduction": "launch 下降",
}

#: 不可行候选的评分。**用 `-1.0` 而不是 `-inf`** —— 前沿排序 / 超体积 / 归一化
#: 都要算均值与标准差，`-inf` 会污染全部下游读数（附录 C 257）。
#: **不可行要「很差」，不要「不是数」。**
INFEASIBLE_SCORE = -1.0


def vector(base: dict[str, Any], cand: dict[str, Any]) -> Point:
    """由两份成本读数算目标向量（全部最大化）。"""
    return (
        1.0 - cand["latency_us"] / base["latency_us"],
        1.0 - cand["peak_bytes"] / base["peak_bytes"],
        1.0 - cand["launches"] / base["launches"],
    )


def reference_point(pad: float) -> Point:
    """超体积参考点：三个目标的下界再外推一点。

    **所有搜索器共用同一个参考点** —— 各跑各的参考点，超体积之间根本没法比
    （附录 C 20）。这里用固定值而不是「数据最小值」正是为了让它是共享的。
    """
    return (-abs(pad), -abs(pad), -abs(pad))


def weights_from(rng_value: float, k: int) -> tuple[float, ...]:
    """ParEGO 式**随机**标量化权重（从单位单纯形上均匀取）。

    固定权重只会找到前沿上的一个点；每一轮重抽才能覆盖整条前沿
    （附录 C「多目标贝叶斯用 ParEGO 式随机标量化」）。
    """
    # 用 `rng_value` 派生 k 个正数再归一化（不引入额外随机源，保持确定性）
    raw = [abs(math.sin((i + 1) * 12.9898 + rng_value * 78.233)) + 1e-6
           for i in range(k)]
    s = sum(raw)
    return tuple(x / s for x in raw)


def tchebycheff(values: Point, weights: Sequence[float], ideal: Point) -> float:
    """加权 Tchebycheff 距离（越小越好）。

    **不用加权和**：加权和到不了非凸前沿上的点。Tchebycheff 取各维
    「离理想点最差缺口」的最大值，非凸段也能被触及。
    """
    return max(w * (ideal[i] - values[i]) for i, w in enumerate(weights))


def scalarize(values: Point, weights: Sequence[float], ideal: Point) -> float:
    """把 Tchebycheff 距离翻成「越大越好」的分数（搜索器内部用）。"""
    return -tchebycheff(values, weights, ideal)


def violation_of(drift: float, tolerance: float, drift_scale: float) -> float:
    """违反量 = `max(0, drift − tolerance) / drift_scale`。

    **除以一个固定的 `drift_scale`（不是除以容差）** —— 这是本项目量纲问题的核心：
    若写成 `drift / tolerance`，τ 越小惩罚项越大，`penalty` 会在 τ→0 时
    **退化成硬约束**，「软约束不保证可行」这条结论就永远暴露不出来
    （同附录 C 244：惩罚项的量纲决定它是不是真的软）。
    """
    return max(0.0, drift - tolerance) / max(drift_scale, 1e-30)


def score_of(feasible: bool, values: Point, *, strategy: str,
             drift: float, tolerance: float, drift_scale: float,
             penalty_weight: float, weights: Sequence[float], ideal: Point) -> float:
    """搜索器看到的分数。**只有这里允许标量化。**"""
    scalar = scalarize(values, weights, ideal)
    if strategy == "reject":
        return scalar if feasible else INFEASIBLE_SCORE
    if strategy == "penalty":
        return scalar - penalty_weight * violation_of(drift, tolerance, drift_scale)
    raise ValueError(f"未知约束策略：{strategy}")


def is_feasible(drift: float, tolerance: float) -> bool:
    """**硬约束的判据就是这一行。** 判据与声明指同一个对象：
    声明说「交出去的答案必须可行」，量的就是答案本身的漂移，不是探索期的。
    """
    return drift <= tolerance
