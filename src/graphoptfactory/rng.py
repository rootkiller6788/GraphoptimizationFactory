"""确定性随机数：只用 `random.Random` + sha256 派生种子。

**禁止内置 `hash()`** —— `PYTHONHASHSEED` 会让它跨进程不稳定，直接破坏可复现性。
所有种子一律走 `derive_seed(*parts)`（sha256 前 8 字节）。

`Rng` **不重排**传入的序列：`choice` / `sample` / `shuffle` 只消费调用方给的顺序。
确定性因此由调用方负责 —— 传进来的必须是**有序且稳定**的序列
（`sorted(...)` 或配置里的 `rule_order`），不能是 `set` / `dict` 的迭代结果：
它们的顺序会随插入顺序与 `PYTHONHASHSEED` 变化，而不重排的话
「同一份输入跑两次」会给出不同的搜索结果。

（第一版文档写的是「`Rng` 先排序再抽取」，与实现不符 —— `shuffle` 排序就等于
不做 shuffle。**契约要写在真正执行它的那一层。**）
"""

from __future__ import annotations

import hashlib
import random
from typing import Iterable, Sequence, TypeVar

T = TypeVar("T")


def derive_seed(*parts: object) -> int:
    """从任意一组可字符串化的成分派生一个 64 位种子。

    成分表就是「什么参数影响什么产物」的声明 —— 多放一个成分，
    就多一条看不见的耦合（`L5-modeloptimizationfactory` 踩过，附录 C 239）。
    """
    raw = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def digest_of(*parts: object) -> str:
    """稳定摘要（前 16 位十六进制）。用于协议指纹 / 图指纹。"""
    raw = "|".join(str(p) for p in parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


class Rng:
    """薄封装：只暴露 `random` / `randint` / `choice` / `shuffle` / `sample`。"""

    __slots__ = ("_r", "seed")

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        self._r = random.Random(self.seed)

    def random(self) -> float:
        return self._r.random()

    def randint(self, a: int, b: int) -> int:
        return self._r.randint(a, b)

    def uniform(self, a: float, b: float) -> float:
        return self._r.uniform(a, b)

    def choice(self, seq: Sequence[T]) -> T:
        """从序列里取一个。序列为空时抛 `ValueError`（不静默返回 None）。"""
        if not seq:
            raise ValueError("choice() 收到空序列")
        return seq[self._r.randrange(len(seq))]

    def shuffle(self, seq: list[T]) -> None:
        self._r.shuffle(seq)

    def sample(self, population: Sequence[T], k: int) -> list[T]:
        return self._r.sample(list(population), k)

    def random_weights(self, n: int, scale: float) -> list[float]:
        """n 个 `U(-scale, scale)` 的权重（用于造确定性张量）。"""
        return [self._r.uniform(-scale, scale) for _ in range(n)]


def sorted_unique(items: Iterable[T]) -> list[T]:
    """排序去重。**凡是拿 set 迭代顺序去驱动输出的地方，都要先过这里。**"""
    return sorted(set(items))
