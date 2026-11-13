"""GraphOptimizationFactory —— 重写计划的搜索闭环（硬约束 = 语义等价）。

**它不写 Kernel，它决定「怎么改图」。** 候选是一条**重写序列**
（有序、不重复），每个候选被真跑一遍，语义等价是可执行的判据而不是声明。

| 层 | 模块 | 职责 |
| --- | --- | --- |
| 世界 | `graph` / `schema` / `evaluator` | 造图、求值、算形状 |
| 变换 | `rewrites` | 8 条合法重写 + 3 条刻意非法对照 |
| 测量 | `cost` / `semantics` | 成本读数（全部解析算出）、漂移与容差 |
| 搜索 | `search/` | oracle / random / greedy / beam / evolution |
| 目标 | `objective` / `pareto` | 三目标、支配、超体积（**结果层不许标量化**） |
| 编排 | `ladder` / `pipeline` | 阶梯（τ × 策略）、产物组装 |
| 交付 | `report` / `emit` / `visualize` / `cli` | 报告、落盘、只读重建、命令 |
"""

from __future__ import annotations

__all__ = ["__version__", "SCHEMA_VERSION"]

__version__ = "1.0.0"

#: 产物格式版本。改变产物字段结构时必须抬它 ——
#: 旧产物被新代码读走时，宁可直接拒绝，也不要静默按新口径解释。
SCHEMA_VERSION = 1
