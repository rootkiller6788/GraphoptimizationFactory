"""`greedy`：每一步选**单步收益最大**的那条规则，不回头。

## 它存在的意义是「被推翻」

设计期望 `greedy_is_myopic` 断言：贪心会比穷举最优差。
**这条期望允许被推翻，而且被推翻才有价值** —— 如果贪心在深度 3 的空间里
就能打平 oracle，那说明「顺序」这件事不重要，本项目的核心命题
（`[fusion, cse]` 与 `[cse, fusion]` 是两个不同的计划）就被削弱了。

## 它烧的预算最少，这一点必须一起报

深度 3、8 条规则：`8 + 7 + 6 = 21` 次评估。`oracle` 是 400 次。
**只报「greedy 分数低」而不报「greedy 只花了 1/20 的预算」，是在抹掉信息。**
所以每个搜索器的 `used` 都进 benchmark，`greedy` 那一行的高分/低分
要连着它的预算一起读。

## 贪心的「单步最优」用的是标量化分数

分数由 `objective.score_of` 给出（固定权重 + 约束策略）。
`reject` 策略下不可行候选得 `-1.0`，于是贪心天然不会往不可行区域迈步 ——
**这是策略在起作用，不是贪心更聪明**。两个策略的贪心结果不同，
正是「约束处理策略」这一轴要测的东西。
"""

from __future__ import annotations

from .base import Evaluation, SearchContext, Visits


class GreedySearcher:
    name = "greedy"

    PARAMS: tuple[str, ...] = ()

    def __init__(self, seed: int = 0) -> None:
        self.seed = int(seed)

    def search(self, ctx: SearchContext, visits: Visits) -> list[Evaluation]:
        out: list[Evaluation] = []
        cur: tuple[str, ...] = ()
        while len(cur) < ctx.max_depth:
            scored: list[tuple[float, str, Evaluation]] = []
            stop = False
            for r in ctx.rule_names:
                if r in cur:
                    continue
                ev = visits(cur + (r,))
                if ev is None:          # 预算耗尽
                    stop = True
                    break
                out.append(ev)
                scored.append((ev.score, r, ev))
            if stop or not scored:
                break
            # 平手时按规则名定序 —— 否则同分候选的胜者取决于 `rule_names`
            # 的迭代顺序，而那是配置里一个「不该影响结果」的细节。
            scored.sort(key=lambda t: (-t[0], t[1]))
            cur = cur + (scored[0][1],)
        return out
