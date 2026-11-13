"""搜索器：`oracle` / `random` / `greedy` / `beam` / `evolution`。

搜索空间 = **有序且不重复的规则序列**，长度 1..`max_depth`。
8 条规则、深度 ≤ 3 ⇒ `P(8,1)+P(8,2)+P(8,3) = 8+56+336 = 400` 个计划 ——
**小到可以穷举**，于是 `oracle` 是真正的全局最优，而不是另一个启发式。
（空间一变大，`oracle` 就退化成一个「更强的搜索器」，那时
`oracle_dominates_all_searchers` 这条验收就不再是数学必然 —— 这条要写进失效边界。）

**顺序是空间的一部分**：`[fusion, cse]` 与 `[cse, fusion]` 是两个不同的计划。
把顺序写进规则里（「先融合再合并」）等于替搜索器做了决定，
而「顺序到底重不重要」正是本项目要测的东西之一。

## 预算是不对称的，而且要写进报告

`oracle` 的预算 = 空间大小（400），其余四个 = `search.budget_per_searcher`（120）。
这个不对称**不是偷懒，是定义**（见 `oracle.py` 模块文档），
但它必须**原样出现在 benchmark 里** —— 藏起来会让读者以为五个搜索器
在同一起跑线上，「谁更强」这个结论就建在沙子上。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .base import (
    EvalCache, Evaluation, PlanResult, SearchContext, Searcher, Visits,
    permutation_counts, sample_plan,
)
from .beam import BeamSearcher
from .evolution import EvolutionSearcher
from .greedy import GreedySearcher
from .oracle import OracleSearcher
from .random_search import RandomSearcher

#: 注册表。`random` 是**必需的对照**，不是凑数 —— 少了它，
#: 「策略的贡献」与「预算的贡献」分不开（附录 C 50 / 搜索闭环约定）。
SEARCHERS: dict[str, type[Searcher]] = {
    "oracle": OracleSearcher,
    "random": RandomSearcher,
    "greedy": GreedySearcher,
    "beam": BeamSearcher,
    "evolution": EvolutionSearcher,
}


def declared_params() -> dict[str, tuple[str, ...]]:
    """每个搜索器声明自己消费 `config.search` 里的哪些键。

    编排层拿它对账：**声明了就必须真的接上**。
    一个写在配置里、却没有任何搜索器读的键，与一个不存在的键
    在输出里长得一模一样（同 `every_threshold_is_wired` 一族）。
    """
    return {name: tuple(getattr(cls, "PARAMS", ())) for name, cls in SEARCHERS.items()}


def make_searcher(name: str, seed: int = 0,
                  params: Mapping[str, Any] | None = None) -> Searcher:
    """按名字造一个搜索器，把 `config.search` 里它声明消费的键喂进去。

    **只喂声明的键** —— 顺手把整个 `search` 段塞进去，会让搜索器
    悄悄依赖上「自己没声明的东西」，于是配置里改一个不相干的键
    会连带改变某个搜索器的行为，而没有任何地方会提示这件事。
    """
    if name not in SEARCHERS:
        raise ValueError(f"未知搜索器：{name}（可选 {sorted(SEARCHERS)}）")
    cls = SEARCHERS[name]
    src = params or {}
    kw = {k: src[k] for k in getattr(cls, "PARAMS", ()) if k in src}
    return cls(seed, **kw)


def budget_for(name: str, ctx: SearchContext, budget_per_searcher: int) -> int:
    """`oracle` 拿整个空间，其余拿统一预算。**这个不对称是显式的、可断言的。**"""
    if getattr(SEARCHERS[name], "full_space", False):
        return ctx.space_size()
    return int(budget_per_searcher)


__all__ = [
    "EvalCache", "Evaluation", "PlanResult", "SearchContext", "Searcher", "Visits",
    "BeamSearcher", "EvolutionSearcher", "GreedySearcher", "OracleSearcher",
    "RandomSearcher", "SEARCHERS", "make_searcher", "budget_for",
    "declared_params", "permutation_counts", "sample_plan",
]
