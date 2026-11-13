"""搜索器协议、两层缓存与带预算的访问器。

## 两层缓存（这是「秒级」的关键）

- **第 1 层（贵）**：`plan → PlanResult`（图 / 成本 / 漂移）。见 `EvalCache`。
  **全进程只算一次** —— 同一个计划在不同容差、不同策略、不同搜索器下被反复提案，
  但「这个计划把图变成什么样」与这些无关。
- **第 2 层（便宜）**：`(plan, strategy, tolerance) → 可行性 + 分数`。纯重导出。

不分层的话，「7 级阶梯 × 2 种策略 × 5 个搜索器」会把同一份图重算 70 遍。

⚠️ **缓存挂在 `EvalCache` 上，不挂在 `SearchContext` 上。**
第一版把 `cache` 放在 `SearchContext.__init__` 里，于是每开一个格子就新建一份缓存 ——
「全进程只算一次」变成了一句写在文档里、实现里不存在的话。

## 去重有两件事，混在一起会让预算失效（这个坑真的踩了）

见 `Visits` 的文档：`oracle` 先跑会把缓存填满。若拿「是否命中缓存」
当「是否花预算」，后面每个搜索器的 `used` 都会是 0 ——
**账面上一分钱没花，却交出了整个空间**，读起来像「随机搜索很强」。
「省算力」与「花预算」是两件事，两套计数器。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

from ..pareto import Point
from ..schema import Plan


@dataclass
class PlanResult:
    """第 1 层：与容差 / 策略无关的读数。"""

    plan: Plan
    drift: float
    values: Point
    metrics: dict[str, Any]
    applied: tuple[str, ...]
    digest: str
    reason: str
    #: 这条计划实际改动了多少个点（各条规则命中次数之和）。
    #: `hits == 0` 的计划与基线**逐位同图**，它出现在前沿上会让
    #: 「搜到的东西」看起来比实际多 —— 所以这个数要报出来，不能只留在脑子里。
    hits: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan.id,
            "rules": list(self.plan.rules),
            "drift": self.drift,
            "values": list(self.values),
            "applied": list(self.applied),
            "digest": self.digest,
            "reason": self.reason,
            "hits": self.hits,
            "metrics": {k: self.metrics[k] for k in sorted(self.metrics)},
        }


@dataclass
class Evaluation:
    """第 2 层：给搜索器看的对象。"""

    result: PlanResult
    feasible: bool
    score: float

    @property
    def plan(self) -> Plan:
        return self.result.plan

    @property
    def values(self) -> Point:
        return self.result.values


class Searcher(Protocol):
    """搜索器。**只准通过 `visits` 看候选** —— 绕过它直接读 `ctx.cache`
    就等于绕过了预算，「谁跑得久」这条读数立刻失去意义。

    类属性（都是可选的声明，注册表与编排层会读它们）：

    - `PARAMS`：这个搜索器消费 `configs/default.json` 的 `search` 段里哪些键。
      **声明了就必须真的接上** —— 编排层会核对，防止「配置里写了但没人读」。
    - `full_space`：它是否需要整个空间才能履行自己的定义（只有 `oracle` 是）。
    """

    name: str

    def search(self, ctx: "SearchContext", visits: "Visits") -> list[Evaluation]:
        ...


class EvalCache:
    """第 1 层缓存：`plan_id -> PlanResult`。**全进程唯一一份。**

    它属于**评估器**，不属于格子 —— 阶梯有 7 级容差 × 2 种策略 × 5 个搜索器
    = 70 个格子，但它们问的是同一个问题「这个计划把图变成什么样」。
    缓存挂在格子上，同一张图会被算 70 遍（`L5-quantfactory` 实测过 44.8s → 0.63s）。

    `unique` / `hits` 是**全局**口径，与 `Visits` 的逐搜索器口径不同：
    前者回答「一共算了几张图」，后者回答「这个搜索器花了多少预算」。
    """

    def __init__(self, evaluate: Callable[[Plan], PlanResult]) -> None:
        self._evaluate = evaluate
        self.store: dict[str, PlanResult] = {}
        self.unique = 0
        self.hits = 0

    def get(self, plan: Plan) -> PlanResult:
        res = self.store.get(plan.id)
        if res is None:
            res = self._evaluate(plan)
            self.store[plan.id] = res
            self.unique += 1
        else:
            self.hits += 1
        return res

    def has(self, plan: Plan) -> bool:
        return plan.id in self.store


class SearchContext:
    """一个 (容差, 策略) 格子里的评估入口。**缓存是外面注入的**（见 `EvalCache`）。"""

    def __init__(self, rule_names: Sequence[str], max_depth: int,
                 cache: EvalCache,
                 score: Callable[[PlanResult], Evaluation],
                 rescore: Callable[[PlanResult, Sequence[float]], float]
                 | None = None) -> None:
        self.rule_names = tuple(rule_names)
        self.max_depth = int(max_depth)
        self.cache = cache
        self._score = score
        self._rescore = rescore

    # -------------------------------------------------------------- 重打分
    def rescore(self, ev: Evaluation, weights: Sequence[float]) -> float:
        """用**搜索器自己的权重**重打分（进化搜索的 ParEGO 用）。

        **只吃第 1 层结果，不重算图** —— 换一组权重是纯算术，
        若这里触发一次图求值，「随机标量化」会贵到不可用。

        没有注入 `rescore` 时退回 `ev.score`：搜索器仍然能跑，
        只是它拿到的是一组固定权重下的分数（**这是一个降级，不是等价**）。
        """
        if self._rescore is None:
            return ev.score
        return self._rescore(ev.result, weights)

    # ------------------------------------------------------------------ 访问
    def is_cached(self, rules: Sequence[str]) -> bool:
        return self.cache.has(Plan(tuple(rules)))

    def evaluate(self, rules: Sequence[str]) -> Evaluation:
        return self._score(self.cache.get(Plan(tuple(rules))))

    # ------------------------------------------------------------------ 空间
    def all_plans(self) -> list[Plan]:
        """整个搜索空间（有序、不重复、长度 1..max_depth）。"""
        out: list[Plan] = []

        def rec(cur: tuple[str, ...]) -> None:
            if cur:
                out.append(Plan(cur))
            if len(cur) >= self.max_depth:
                return
            for r in self.rule_names:
                if r not in cur:
                    rec(cur + (r,))

        rec(())
        return out

    def space_size(self) -> int:
        n, d = len(self.rule_names), self.max_depth
        total, perm = 0, 1
        for k in range(1, d + 1):
            perm = perm * (n - k + 1)
            total += perm
        return total


class Visits:
    """带预算的访问器。

    ## 这里有两个**不同**的去重问题，混在一起会让预算失效

    第一版只看了 `ctx.is_cached`，于是：

    - `oracle` 先跑，400 个计划全进全局缓存；
    - 后面的 `random` 每次提案都是缓存命中 ⇒ `used` 恒为 0，
      **它交出了整个空间，而账面上一次预算都没花**。

    症状看起来像「随机搜索很强」。真因是：**「省算力」与「花预算」不是同一件事。**

    | 计数器 | 含义 | 该不该计预算 |
    | --- | --- | --- |
    | `used` | 本搜索器**首次**提案的唯一计划 | **计** —— 这才是搜索预算 |
    | `hits` | 本搜索器重复提案同一个计划 | 不计（重复不是搜索的贡献） |
    | `shared` | 提案已被**别的**搜索器评估过 | 不计（省的是算力，不是预算） |

    所以 `Visits` 自己维护一个 `_seen`，**按「本搜索器提案过什么」去重**；
    全局缓存的去重是另一件事，它只负责别把同一张图算两遍。

    ## 后跑的搜索器白捡答案，这公平吗

    **公平，但必须说清是哪一种公平。** 搜索器的提案序列**不依赖**缓存内容
    （命中只是把同一个读数再取出来一次），所以「有缓存」与「冷缓存」跑出的
    **结果逐位相同**，只有墙钟时间不同。⇒ 比较结论不受污染，
    但 `shared` 这个数要报出来 —— 否则「5 个搜索器只花了 1.7 秒」
    会被读成「搜索很便宜」，而真相是「第 1 个搜索器付了钱，后 4 个搭了车」。
    """

    def __init__(self, ctx: SearchContext, budget: int) -> None:
        self.ctx = ctx
        self.budget = int(budget)
        self.used = 0
        self.hits = 0
        self.shared = 0
        self._seen: set[tuple[str, ...]] = set()

    def __call__(self, rules: Sequence[str]) -> Evaluation | None:
        key = tuple(rules)
        if key in self._seen:              # 自己重复提案：免费，但不计预算
            self.hits += 1
            return self.ctx.evaluate(key)
        if self.used >= self.budget:       # 预算用完：**返回 None**，不静默降级
            return None
        self.used += 1
        self._seen.add(key)
        if self.ctx.is_cached(key):
            self.shared += 1
        return self.ctx.evaluate(key)

    @property
    def exhausted(self) -> bool:
        return self.used >= self.budget

    def to_dict(self) -> dict[str, Any]:
        return {
            "budget": self.budget,
            "used": self.used,
            "hits": self.hits,
            "shared": self.shared,
        }


# --------------------------------------------------------------------- 空间采样
def permutation_counts(n: int, d: int) -> list[int]:
    """各深度上的计划条数：`[P(n,1), P(n,2), ..., P(n,d)]`。"""
    counts: list[int] = []
    perm = 1
    for k in range(1, d + 1):
        perm *= (n - k + 1)
        counts.append(perm)
    return counts


def sample_plan(rng: Rng, rule_names: Sequence[str], max_depth: int) -> tuple[str, ...]:
    """**在空间上均匀**抽一个计划。

    不是「均匀抽深度、再均匀抽规则」—— 那样深度 1 会被抽到 1/d 的概率，
    而它在空间里只占 `P(n,1)/Σ` 。本项目 `n=8, d=3`：深度 3 占 336/400。
    均匀抽深度会让随机对照**过度采样浅计划**，于是「随机搜不到好解」
    变成抽样偏置的产物，而不是搜索能力的读数（同附录 C 20 一族：
    读数被一个没人看的实现细节决定）。
    """
    counts = permutation_counts(len(rule_names), max_depth)
    x = rng.random() * sum(counts)
    depth = len(counts)
    acc = 0
    for k, c in enumerate(counts, start=1):
        acc += c
        if x < acc:
            depth = k
            break
    return tuple(rng.sample(rule_names, depth))
