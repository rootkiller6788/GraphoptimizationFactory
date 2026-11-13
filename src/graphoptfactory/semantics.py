"""语义校验：把基线图与候选图**都跑一遍**，比输出。

设计稿里那句「`Validate Semantics` 是不可省略的一环」在这里落成一句可执行的话：

    两张图喂同一批探针，输出之差超过容差 ⇒ 候选不可行。

三件事必须分清（否则「校验器」会退化成同义反复）：

1. **容差是硬约束**，不是评分项。`feasible=False` 的候选**不参与前沿**。
2. **漂移必须有一个来源**：合法但非逐位等价的重写（浮点重结合）会带来
   1e-16 量级的漂移；非法重写会带来 1e-2 量级。**两者之间必须有一条缝**，
   而这条缝有多宽是本项目的核心读数 —— 见 `ladder.py`。
3. **校验器本身必须被验。** 一个「永远返回 True」的校验器与没有校验器在输出里
   长得一模一样（附录 C 261/275）—— 所以 `adversarial.py` 会注入**刻意非法**的重写，
   并断言它们**被拒**。拒掉几条这个数要**算出来并落盘**。

⚠️ **`NaN` 必须显式判非法**：`abs(nan - x) > tol` 永远是 `False`（附录 C 9/171），
靠容差比较**永远**判不出 NaN。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from .evaluator import has_nan, run_outputs
from .schema import Graph, GraphError, Tensor


@dataclass
class Validation:
    """一次校验的完整读数。**通过也要留证据**，否则「为什么留下」答不上来。"""

    ok: bool
    drift: float
    tolerance: float
    n_probe: int
    reason: str
    per_probe: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "drift": self.drift,
            "tolerance": self.tolerance,
            "n_probe": self.n_probe,
            "reason": self.reason,
            "per_probe": [round(x, 18) for x in self.per_probe],
        }


def _max_abs(t: Tensor) -> float:
    return max((abs(x) for row in t for x in row), default=0.0)


def output_drift(a: Sequence[Tensor], b: Sequence[Tensor]) -> float:
    """**尺度无关**的漂移：`max|a−b| / max(1, max|a|)`。

    不除尺度的话，「漂移」会随激活幅度线性变化 —— 那样同一个重写在不同的
    权重幅度下会得到不同的容差判定，而容差是**配置里的一个数**，它不该随世界漂移。
    """
    if len(a) != len(b):
        raise GraphError(f"输出个数不同：{len(a)} vs {len(b)}")
    worst = 0.0
    for ta, tb in zip(a, b):
        if len(ta) != len(tb) or (ta and len(ta[0]) != len(tb[0])):
            raise GraphError(f"输出形状不同：{(len(ta), len(ta[0]))} vs "
                             f"{(len(tb), len(tb[0]))}")
        scale = max(1.0, _max_abs(ta))
        for ra, rb in zip(ta, tb):
            for x, y in zip(ra, rb):
                d = abs(x - y) / scale
                if d > worst:
                    worst = d
    return worst


def validate(baseline: Graph, candidate: Graph, probes: Sequence[Tensor],
             tolerance: float) -> Validation:
    """跑两张图并比对。**遇到形状不匹配 / NaN 直接判非法**，不靠容差兜住。"""
    per: list[float] = []
    for k, feed in enumerate(probes):
        try:
            a = run_outputs(baseline, feed)
            b = run_outputs(candidate, feed)
        except GraphError as e:                     # 形状错 = 非法，且要说清原因
            return Validation(False, float("inf"), tolerance, len(probes),
                              f"探针 {k} 求值失败：{e}", tuple(per))
        if any(has_nan(t) for t in b):
            return Validation(False, float("inf"), tolerance, len(probes),
                              f"探针 {k} 的输出含 NaN（NaN 不走容差比较）",
                              tuple(per))
        try:
            per.append(output_drift(a, b))
        except GraphError as e:
            return Validation(False, float("inf"), tolerance, len(probes),
                              f"探针 {k} 输出形状不同：{e}", tuple(per))
    drift = max(per) if per else 0.0
    ok = drift <= tolerance
    reason = "" if ok else f"漂移 {drift:.3e} > 容差 {tolerance:.3e}"
    return Validation(ok, drift, tolerance, len(probes), reason, tuple(per))


def is_bit_identical(baseline: Graph, candidate: Graph, probes: Sequence[Tensor]) -> bool:
    """逐位相同（漂移**恰好** 0.0）。用来把「精确等价的重写」与「近似等价的重写」分开。

    这个区分不是学术洁癖：它决定了 τ=0 那一级的可行域里有什么。
    """
    return validate(baseline, candidate, probes, 0.0).ok


def tolerance_grid(baseline: Graph, variants: dict[str, Graph],
                   probes: Sequence[Tensor],
                   tolerances: Sequence[float]) -> dict[str, list[bool]]:
    """`变体 × 容差` 的接受矩阵 —— 容差崖就是在这张表上「看见」的。

    行是变体（合法 / 非法），列是容差。**非法变体在任何一列被接受 = 校验器漏了**。
    """
    out: dict[str, list[bool]] = {}
    for name, g in variants.items():
        row: list[bool] = []
        for tol in tolerances:
            row.append(validate(baseline, g, probes, tol).ok)
        out[name] = row
    return out
