"""`evolution`：种群 + 交叉 + 变异，用 **ParEGO 式随机标量化**做适应度。

## 为什么适应度权重每一代都要重抽

固定一组权重只会把种群往**前沿上的一个点**推 —— 那是单目标优化，
不是多目标。每代给每个个体抽一组 `U(单纯形)` 权重，
等价于每代换一个「这次最看重哪一维」的立场，种群才会沿着整条前沿铺开
（`objective.weights_from` 就是干这个的）。

重打分走 `ctx.rescore`，**只吃第 1 层结果**：换权重是纯算术。
若换权重要重算图，「随机标量化」会贵到用不起，于是这条设计会被
实现悄悄降级成固定权重 —— 而输出里看不出来（附录 C 一族）。

## 基因组 = 有序不重复的规则序列

长度 `1..max_depth`，规则不重复。**长度是基因组的一部分**：
一个深度 1 的基因组与深度 3 的基因组可以同台竞争，不需要先补齐深度。
补齐会让「这条规则有用吗」与「深度够不够」混成一个数。

## 交叉要保序、保不重复

子代 = 父甲的前缀 + 父乙里还没出现过的规则（按乙的顺序），截到 `max_depth`。
**不能用「逐位随机挑一个」**：那会产生重复规则，而重复规则的计划
在空间里根本不存在（`Plan` 是有序不重复序列），于是搜索器会持续
提案一批永远评估不到、白白烧预算的候选。

## 确定性

所有随机走 `Rng`（sha256 派生种子）。**种群初始化也用 `sample_plan`**，
与 `random` 搜索器同一份抽样口径 —— 否则「进化 vs 随机」的差异里
会混进「两者初始种群分布不同」。
"""

from __future__ import annotations

from ..objective import weights_from
from ..rng import Rng, derive_seed
from .base import Evaluation, SearchContext, Visits, sample_plan


def _crossover(rng: Rng, a: tuple[str, ...], b: tuple[str, ...],
               rule_names: tuple[str, ...], max_depth: int) -> tuple[str, ...]:
    cut = rng.randint(1, len(a))
    child = list(a[:cut])
    for r in b:
        if len(child) >= max_depth:
            break
        if r not in child:
            child.append(r)
    if not child:                      # a 非空 ⇒ 不会发生；留一道保险
        child = [rng.choice(rule_names)]
    return tuple(child)


def _mutate(rng: Rng, g: tuple[str, ...], rule_names: tuple[str, ...],
            max_depth: int) -> tuple[str, ...]:
    """三种变异：加一条 / 去一条 / 换一条。**保证结果仍是有序不重复序列。**"""
    unused = [r for r in rule_names if r not in g]
    op = rng.randint(0, 2)
    if op == 0 and len(g) < max_depth and unused:
        pos = rng.randint(0, len(g))
        return tuple(list(g[:pos]) + [rng.choice(unused)] + list(g[pos:]))
    if op == 1 and len(g) > 1:
        pos = rng.randint(0, len(g) - 1)
        return tuple(g[:pos] + g[pos + 1:])
    if op == 2 and unused:
        pos = rng.randint(0, len(g) - 1)
        return tuple(g[:pos] + (rng.choice(unused),) + g[pos + 1:])
    return g


class EvolutionSearcher:
    name = "evolution"

    PARAMS = ("population", "mutation_rate", "max_moves")

    def __init__(self, seed: int = 0, population: int = 12,
                 mutation_rate: float = 0.35, max_moves: int = 24) -> None:
        self.seed = int(seed)
        self.population = max(2, int(population))
        self.mutation_rate = float(mutation_rate)
        self.max_moves = max(1, int(max_moves))

    # ------------------------------------------------------------------ 主循环
    def search(self, ctx: SearchContext, visits: Visits) -> list[Evaluation]:
        rng = Rng(derive_seed(self.seed, self.name, ctx.max_depth,
                              len(ctx.rule_names), self.population,
                              self.mutation_rate, self.max_moves))
        out: list[Evaluation] = []

        # ---- 初始种群（与 random 同一抽样口径）
        genomes: list[tuple[str, ...]] = []
        seen: set[tuple[str, ...]] = set()
        while len(genomes) < self.population:
            g = sample_plan(rng, ctx.rule_names, ctx.max_depth)
            if g in seen:
                continue
            seen.add(g)
            genomes.append(g)

        for _ in range(self.max_moves):
            # ---- 评估（`visits` 返回 None ⇒ 预算耗尽，立刻停，不吞）
            scored: list[tuple[float, tuple[str, ...], Evaluation]] = []
            exhausted = False
            for g in genomes:
                ev = visits(g)
                if ev is None:
                    exhausted = True
                    break
                out.append(ev)
                w = weights_from(rng.random(), len(ev.values))
                scored.append((ctx.rescore(ev, w), g, ev))
            if exhausted or not scored:
                break

            # ---- 选择（精英一半，平手按基因组定序，避免迭代顺序渗入）
            scored.sort(key=lambda t: (-t[0], t[1]))
            elite = [g for _, g, _ in scored[: max(1, len(scored) // 2)]]

            # ---- 繁殖
            children: list[tuple[str, ...]] = []
            guard = 0
            while len(children) < self.population - len(elite) and guard < 256:
                guard += 1
                a = rng.choice(elite)
                b = rng.choice(elite)
                child = _crossover(rng, a, b, ctx.rule_names, ctx.max_depth)
                if rng.random() < self.mutation_rate:
                    child = _mutate(rng, child, ctx.rule_names, ctx.max_depth)
                if child in seen:
                    continue
                seen.add(child)
                children.append(child)
            if not children:
                break
            genomes = elite + children
        return out
