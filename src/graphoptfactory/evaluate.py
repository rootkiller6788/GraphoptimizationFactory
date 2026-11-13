"""候选的**第 1 层**评估：计划 → 图 → 读数。**与容差 / 策略无关。**

这一层贵的只有图求值：每个计划要把重写序列真的施加一遍、把图真的跑一遍。
它不进容差循环，也不进策略循环 —— 换一个 τ 或换一个策略，
「这个计划把图变成什么样」不会变（`search/base.py` 模块文档里的两层缓存）。

## 结构失败要**记下来并计数**，不是吞掉

有些计划会在施加过程中抛 `GraphError`（形状对不上、父节点不存在）。
**这个不能静默变成「这条规则不适用、0 命中」** ——
那正是 `rewrites.py` 里删掉的那个 `except GraphError` 的同一件事：
一个吞掉异常的兜底等于一条静默失效的通道。

这里的处理是第三条路：**记录 + 计数 + 上报**。

- 记成一个 `PlanResult`，`drift=inf`、`reason` 写清原因；
- 计数器 `n_structural` / `structural_reasons` 原样进 `out/`；
- 验收门槛 `max_structural_failures = 0` 断言它是 0。

于是「有它在兜着」这件事本身变成一条被断言的读数：
真出现结构失败会红，不会伪装成「搜索器表现一般」。
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from .cost import metrics
from .graph import shape_table
from .objective import vector
from .rewrites import apply_rule
from .schema import Graph, GraphError, Plan, Tensor
from .semantics import validate
from .search.base import PlanResult

#: 结构失败时的目标向量。**不用 `inf` / `nan`** —— 它们会污染均值与归一化
#: （坑 257 的同一族）。反正不可行候选不参与前沿，取什么都不影响结论。
_ZERO_VECTOR = (0.0, 0.0, 0.0)


class PlanEvaluator:
    """`Plan -> PlanResult`。可调用对象，直接当 `SearchContext` 的 `evaluate` 用。"""

    def __init__(self, cfg: dict[str, Any], baseline: Graph,
                 probes: Sequence[Tensor], base_metrics: dict[str, Any]) -> None:
        self.cfg = cfg
        self.baseline = baseline
        self.probes = tuple(probes)
        self.base_metrics = base_metrics
        #: 结构失败的计数与原因分布（**都要落盘**，见模块文档）。
        self.n_structural = 0
        self.structural_reasons: dict[str, int] = {}
        #: 逐位等价 / 近似等价的计划数（在层 1 就能分，因为漂移与容差无关）。
        self.n_bit_identical = 0
        self.n_noop = 0

    # ------------------------------------------------------------------ 施加
    def _apply(self, plan: Plan) -> tuple[Graph, int]:
        g = self.baseline
        hits = 0
        for name in plan.rules:
            g, h = apply_rule(g, name, self.cfg)
            hits += int(h)
        return g, hits

    # ------------------------------------------------------------------ 评估
    def __call__(self, plan: Plan) -> PlanResult:
        try:
            g, hits = self._apply(plan)
            shapes = shape_table(g, self.cfg)
            m = metrics(g, shapes, self.cfg)
        except GraphError as e:
            self.n_structural += 1
            key = f"{type(e).__name__}: {e}"[:120]
            self.structural_reasons[key] = self.structural_reasons.get(key, 0) + 1
            return PlanResult(plan, float("inf"), _ZERO_VECTOR, {},
                              tuple(plan.rules), "", f"结构失败：{e}", hits=0)

        # 容差传 0.0：**只取漂移这个与容差无关的量**。
        # 层 1 不该知道任何容差（知道了就会有人拿它当判据）。
        v = validate(self.baseline, g, self.probes, 0.0)
        reason = v.reason if not math.isfinite(v.drift) else ""
        if v.drift == 0.0:
            self.n_bit_identical += 1
        if hits == 0:
            self.n_noop += 1

        return PlanResult(
            plan=plan,
            drift=v.drift,
            values=vector(self.base_metrics, m),
            metrics=m,
            applied=tuple(plan.rules),
            digest=g.digest(),
            reason=reason,
            hits=hits,
        )

    # ------------------------------------------------------------------ 诊断
    def diagnostics(self) -> dict[str, Any]:
        return {
            "n_structural": self.n_structural,
            "structural_reasons": {k: self.structural_reasons[k]
                                   for k in sorted(self.structural_reasons)},
            "n_bit_identical": self.n_bit_identical,
            "n_noop": self.n_noop,
        }
