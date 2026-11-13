"""`beam`：每层保留 `beam_width` 个最好的前缀，再往后展开。

## 与 `greedy` 的差别只有一个数：宽度

`beam_width=1` 时它**退化成贪心**（每一步只留一个前缀）。这不是巧合，
是本项目刻意留的一条**可对账关系**：两个搜索器在同一张图、同一预算下的
读数必须在 `width=1` 时逐位一致。不一致就说明「保留前缀」这件事
在两份实现里不是同一件事 —— 那种 bug 不会红，只会让两条曲线看起来
像两个独立的算法（同 `L5-modeloptimizationfactory` 的消融纪律：
一个算法是另一个的消融时，它必须只差一处，而且要被机器核对）。

所以 `PARAMS` 里那个 `beam_width` 不是一个普通超参，
它是这条对账关系的旋钮。

## 预算截断要**从中间断**

展开到一半预算没了：保留已经评估完的孩子，丢掉没评估的。
不要「先全部提案再统一评估」—— 那样会在预算耗尽时
返回一批**没有读数**的候选，而下游会当成「它们很差」。
"""

from __future__ import annotations

from .base import Evaluation, SearchContext, Visits


class BeamSearcher:
    name = "beam"

    PARAMS = ("beam_width",)

    def __init__(self, seed: int = 0, beam_width: int = 4) -> None:
        self.seed = int(seed)
        self.beam_width = max(1, int(beam_width))

    def search(self, ctx: SearchContext, visits: Visits) -> list[Evaluation]:
        out: list[Evaluation] = []
        # 前缀 → 它的读数。空计划没有读数（基线不是候选），
        # 所以第一层直接展开成深度 1。
        frontier: list[Evaluation] = []
        for r in ctx.rule_names:
            ev = visits((r,))
            if ev is None:
                return out
            out.append(ev)
            frontier.append(ev)

        for _ in range(2, ctx.max_depth + 1):
            children: list[Evaluation] = []
            stop = False
            for parent in frontier:
                used = set(parent.plan.rules)
                for r in ctx.rule_names:
                    if r in used:
                        continue
                    ev = visits(parent.plan.rules + (r,))
                    if ev is None:
                        stop = True
                        break
                    out.append(ev)
                    children.append(ev)
                if stop:
                    break
            if stop or not children:
                break
            children.sort(key=lambda e: (-e.score, e.plan.rules))
            frontier = children[: self.beam_width]
        return out
