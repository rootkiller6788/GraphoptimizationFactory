"""`oracle`：穷举整个搜索空间。

**它不是「第六个搜索器」，它是定义。** 空间只有 400 个计划（`n=8, d=3`），
所以「全局最优」在这里是可计算的，不是估计的。于是
`oracle_dominates_all_searchers` 这条验收是**数学必然**，而不是运气好。

## 为什么它的预算必须等于空间大小

若把 `oracle` 也压在 `budget_per_searcher=120` 下，它就跑不完 400 个计划 ——
那时它交出的是「120 个计划里的最优」，**它就不再是 oracle 而是第六个启发式**，
而 `oracle_dominates_all_searchers` 也从「必然」退化成「可能」，
一条本来恒真的判据会悄悄变成一个会红的判据。两者在输出里长得一模一样。

⇒ 编排层按 `full_space` 这个声明给 `oracle` 抬高预算，
并且**把「预算不对称」原样写进 benchmark**。藏起来的话，
读者会以为五个搜索器在同一起跑线上，而「谁更强」这个结论就建在沙子上。
"""

from __future__ import annotations

from .base import Evaluation, SearchContext, Visits


class OracleSearcher:
    name = "oracle"

    #: 只有它需要整个空间（见模块文档：这不是特权，是定义）。
    full_space = True

    #: 不吃任何搜索超参 —— 它没有旋钮。**这条声明本身就是读数**：
    #: 一个「没有超参的参照物」才配当参照物。
    PARAMS: tuple[str, ...] = ()

    def __init__(self, seed: int = 0) -> None:
        self.seed = int(seed)

    def search(self, ctx: SearchContext, visits: Visits) -> list[Evaluation]:
        out: list[Evaluation] = []
        for plan in ctx.all_plans():
            ev = visits(plan.rules)
            if ev is None:
                # 预算被截断了 —— 这是配置错误，不是搜索结果。
                # **必须响**：静默返回一个不完整的前沿，会伪装成
                # 「oracle 也就这样」，把一条恒真判据变成假结论（附录 C 245 一族）。
                raise RuntimeError(
                    f"oracle 预算不足：空间 {ctx.space_size()} 个计划，"
                    f"只走到第 {len(out)} 个。oracle 的预算必须等于空间大小。"
                )
            out.append(ev)
        if len(out) != ctx.space_size():
            raise RuntimeError(
                f"oracle 覆盖不全：{len(out)} != {ctx.space_size()}"
            )
        return out
