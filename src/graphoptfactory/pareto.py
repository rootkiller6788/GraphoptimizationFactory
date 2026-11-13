"""Pareto 工具：支配 / 非支配排序 / 拥挤度 / 超体积 / 间距 / 相关矩阵。

**坐标约定只有一套**：全部**最大化**。混用两套约定会静默削掉半个前沿
（附录 C 18）—— 所以 `Reading.vector()` 一律返回「越大越好」的形式，
代价类量在目标层就已经取过负号了。
"""

from __future__ import annotations

import math
from typing import Sequence

Point = tuple[float, ...]


def dominates(a: Point, b: Point) -> bool:
    """`a` 支配 `b`：**全部 ≥ 且至少一个 >**。"""
    ge = all(x >= y for x, y in zip(a, b))
    gt = any(x > y for x, y in zip(a, b))
    return ge and gt


def non_dominated_indices(points: Sequence[Point]) -> list[int]:
    """返回**下标**（不是点）—— 调用方常常只传子集进来，
    返回点会让「按位置索引」这件事失去意义（附录 C 16）。"""
    out: list[int] = []
    for i, p in enumerate(points):
        if not any(dominates(q, p) for j, q in enumerate(points) if j != i):
            out.append(i)
    return out


def fast_non_dominated_sort(points: Sequence[Point]) -> list[list[int]]:
    """NSGA-II 的分层：返回每一层的下标列表，第 0 层就是前沿。"""
    n = len(points)
    dominated_by: list[list[int]] = [[] for _ in range(n)]
    count = [0] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(points[i], points[j]):
                dominated_by[i].append(j)
            elif dominates(points[j], points[i]):
                count[i] += 1
    layers: list[list[int]] = []
    current = [i for i in range(n) if count[i] == 0]
    while current:
        layers.append(sorted(current))
        nxt: list[int] = []
        for i in current:
            for j in dominated_by[i]:
                count[j] -= 1
                if count[j] == 0:
                    nxt.append(j)
        current = nxt
    return layers


def crowding_distance(points: Sequence[Point]) -> list[float]:
    """拥挤度。**按位置返回 `list[float]`**，由调用方自己 `zip` 回原下标
    （附录 C 16：返回 dict 会在「只传子集」时 `IndexError`）。"""
    n = len(points)
    if n == 0:
        return []
    if n <= 2:
        return [float("inf")] * n
    dim = len(points[0])
    dist = [0.0] * n
    for d in range(dim):
        order = sorted(range(n), key=lambda i: points[i][d])
        lo, hi = points[order[0]][d], points[order[-1]][d]
        dist[order[0]] = float("inf")
        dist[order[-1]] = float("inf")
        span = hi - lo
        if span <= 0:
            continue
        for k in range(1, n - 1):
            i = order[k]
            dist[i] += (points[order[k + 1]][d] - points[order[k - 1]][d]) / span
    return dist


def hypervolume(points: Sequence[Point], ref: Point) -> float:
    """超体积（全部最大化，`ref` 被所有点支配）。递归切片（HSO）。

    这是对外唯一的单一数字，**必须用独立可验的算例钉住**（附录 C 55）：
    单点、两点、加被支配点不变、参考点反向支配时归零。四条都在 `adversarial` 里。
    """
    d = len(ref)
    pts = [tuple(p) for p in points if all(p[i] > ref[i] for i in range(d))]
    if not pts:
        return 0.0
    if d == 1:
        return max(p[0] for p in pts) - ref[0]
    pts.sort(key=lambda p: -p[0])
    vol = 0.0
    for i, p in enumerate(pts):
        lower = pts[i + 1][0] if i + 1 < len(pts) else ref[0]
        width = p[0] - lower
        if width <= 0:
            continue
        cross = [q[1:] for q in pts[: i + 1]]
        vol += width * hypervolume(cross, ref[1:])
    return vol


def spacing(points: Sequence[Point]) -> float:
    """间距。**必须先按维归一化** —— 不归一化时结果完全由量纲最大的那一维决定，
    而且不报错，只会给出一个看着像模像样的数字（附录 C 17）。"""
    n = len(points)
    if n < 2:
        return 0.0
    dim = len(points[0])
    lo = [min(p[d] for p in points) for d in range(dim)]
    hi = [max(p[d] for p in points) for d in range(dim)]
    span = [hi[d] - lo[d] for d in range(dim)]

    def norm(p: Point) -> Point:
        return tuple((p[d] - lo[d]) / span[d] if span[d] > 0 else 0.0
                     for d in range(dim))

    npts = [norm(p) for p in points]
    dists: list[float] = []
    for i in range(n):
        best = min(math.dist(npts[i], npts[j]) for j in range(n) if j != i)
        dists.append(best)
    mean = sum(dists) / len(dists)
    var = sum((x - mean) ** 2 for x in dists) / len(dists)
    return math.sqrt(var)


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """秩相关。**样本不足返回 `nan`**（同 `pearson`：缺失的证据 ≠ 否定的证据）。

    用秩而不是值，是因为要问的问题常常是「排序对不对」而不是「值准不准」：
    「解析延迟能不能预测实际耗时」这个问题里，解析延迟差一个常数倍
    完全不影响结论，但它会让皮尔逊相关系数变脸。
    """
    n = len(a)
    if n < 3 or len(b) != n:
        return float("nan")

    def ranks(xs: Sequence[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: xs[i])
        out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0          # 并列取平均秩
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    return pearson(ranks(a), ranks(b))


def pearson(a: Sequence[float], b: Sequence[float]) -> float:
    """皮尔逊相关系数。**样本不足或方差为 0 时返回 `nan` 而不是 0** ——
    返回 0 会让「算不出来」伪装成「不相关」（附录 C 184：缺失的证据 ≠ 否定的证据）。"""
    n = len(a)
    if n < 3 or len(b) != n:
        return float("nan")
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    if va <= 0 or vb <= 0:
        return float("nan")
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    return cov / math.sqrt(va * vb)


def correlation_matrix(points: Sequence[Point], names: Sequence[str]
                       ) -> dict[str, dict[str, float]]:
    """两两相关矩阵（含对角线 = 1.0，非对角可能为 `nan`）。"""
    out: dict[str, dict[str, float]] = {}
    for i, ni in enumerate(names):
        row: dict[str, float] = {}
        for j, nj in enumerate(names):
            row[nj] = 1.0 if i == j else pearson([p[i] for p in points],
                                                 [p[j] for p in points])
        out[ni] = row
    return out
