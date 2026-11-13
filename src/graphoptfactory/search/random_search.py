"""`random`：在空间上均匀抽计划。**这是必需的对照，不是凑数的第六行。**

少了它，「策略的贡献」与「预算的贡献」分不开：`beam` 比 `greedy` 好，
可能是因为 beam 更聪明，也可能只是因为它评估了更多候选。
`random` 拿同样的预算、同样的缓存，唯一没有的就是「策略」。

**文件名叫 `random_search.py` 而不是 `random.py`** —— 后者会在包内
遮蔽标准库 `random`，而 `rng.py` 正是靠标准库做确定性随机的。
遮蔽不会报错，只会让某些导入路径下拿到错误的模块（附录 C 一族：
「名字撞车不报错」）。

抽样口径见 `base.sample_plan`：**在空间上均匀**，不是均匀抽深度。
"""

from __future__ import annotations

from ..rng import Rng, derive_seed
from .base import Evaluation, SearchContext, Visits, sample_plan


class RandomSearcher:
    name = "random"

    #: 它没有超参：均匀抽样就是它的全部定义。
    PARAMS: tuple[str, ...] = ()

    def __init__(self, seed: int = 0) -> None:
        self.seed = int(seed)

    def search(self, ctx: SearchContext, visits: Visits) -> list[Evaluation]:
        rng = Rng(derive_seed(self.seed, self.name, ctx.max_depth,
                              len(ctx.rule_names)))
        out: list[Evaluation] = []
        seen: set[tuple[str, ...]] = set()
        # 重复提案会白白烧掉循环次数；给一个与预算成比例的兜底，
        # 保证「抽不出新计划」时能退出（空间有限，重复必然越来越多）。
        attempts = 0
        cap = visits.budget * 40 + 128
        while not visits.exhausted and attempts < cap:
            attempts += 1
            rules = sample_plan(rng, ctx.rule_names, ctx.max_depth)
            if rules in seen:
                continue
            seen.add(rules)
            ev = visits(rules)
            if ev is None:
                break
            out.append(ev)
        return out
