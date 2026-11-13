"""纯 Python 求值器：**真的把图跑一遍**，不是手写打分公式。

这是本项目最硬的一件事：**语义等价是可执行的**。不需要相信任何人的声明，
把两张图喂同一批探针，比输出即可。

性能上先用标准库里的 C 实现：`math.sumprod`（CPython 3.12+）替代手写内积
（`ArchitectureFactory` 实测前向 2.6× / 反向 2.3×，见附录 C 52）。
`sum(x*y for ...)` 反而更慢（生成器开销）。
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from .schema import Graph, GraphError, Tensor

#: 求值器里用的算子集合必须与 `schema.OPS` 一致 —— 少一个就会在运行时才炸。
_MATMUL_LIKE = ("matmul", "fused_mm_relu", "fused_mm_bias_relu")


def _matmul(a: Tensor, b: Tensor) -> Tensor:
    """`a @ b`。内积走 `math.sumprod`（C 实现）。"""
    n, k = len(a), len(a[0])
    k2, m = len(b), len(b[0])
    if k != k2:
        raise GraphError(f"matmul 内维不匹配：{k} vs {k2}")
    # 列优先取列，避免 g[i][j] 跨行索引破坏局部性（附录 C 52）
    cols = [tuple(b[r][j] for r in range(k)) for j in range(m)]
    return tuple(tuple(math.sumprod(a[i], cols[j]) for j in range(m))
                 for i in range(n))


def _broadcast_binop(a: Tensor, b: Tensor, op: str) -> Tensor:
    ra, ca = len(a), len(a[0])
    rb, cb = len(b), len(b[0])
    if (ra, ca) == (rb, cb):
        if op == "add":
            return tuple(tuple(x + y for x, y in zip(row_a, row_b))
                         for row_a, row_b in zip(a, b))
        return tuple(tuple(x * y for x, y in zip(row_a, row_b))
                     for row_a, row_b in zip(a, b))
    if ca == cb and (ra == 1 or rb == 1):
        rows, cols = max(ra, rb), ca
        if op == "add":
            return tuple(tuple((a[i % ra][j] + b[i % rb][j]) for j in range(cols))
                         for i in range(rows))
        return tuple(tuple((a[i % ra][j] * b[i % rb][j]) for j in range(cols))
                     for i in range(rows))
    raise GraphError(f"{op} 形状不可广播：{(ra, ca)} vs {(rb, cb)}")

def evaluate(g: Graph, feed: Tensor) -> dict[str, Tensor]:
    """按拓扑序求值，返回**逐节点**的值表。

    返回逐节点值而不是只返回输出，是为了让诊断（「是哪一步炸的」）有东西可看 ——
    只返回输出的求值器在出错时只能告诉你「结果不对」。
    """
    if len(g.inputs) != 1:
        raise GraphError(f"本项目的图恰好一个输入，实得 {len(g.inputs)} 个")
    values: dict[str, Tensor] = {g.inputs[0]: feed}

    for nid in g.topo():
        node = g.nodes[nid]
        op = node.op
        if op == "input":
            continue
        if op == "const":
            values[nid] = g.consts[nid]
            continue
        ins = [values[i] for i in node.inputs]
        if op == "matmul":
            values[nid] = _matmul(ins[0], ins[1])
        elif op == "fused_mm_relu":
            mm = _matmul(ins[0], ins[1])
            values[nid] = tuple(tuple(max(0.0, v) for v in row) for row in mm)
        elif op == "fused_mm_bias_relu":
            mm = _matmul(ins[0], ins[1])
            bias = ins[2][0]
            values[nid] = tuple(tuple(max(0.0, v + bias[j]) for j, v in enumerate(row))
                                for row in mm)
        elif op in ("add", "mul"):
            values[nid] = _broadcast_binop(ins[0], ins[1], op)
        elif op == "relu":
            values[nid] = tuple(tuple(max(0.0, v) for v in row) for row in ins[0])
        elif op == "transpose":
            src = ins[0]
            values[nid] = tuple(tuple(src[i][j] for i in range(len(src)))
                                for j in range(len(src[0])))
        elif op == "scale":
            s = float(node.attr("s", 1.0))
            values[nid] = tuple(tuple(v * s for v in row) for row in ins[0])
        else:  # pragma: no cover - OPS 已封口
            raise GraphError(f"{nid}: 求值器不认识算子 {op}")
    return values


def run_outputs(g: Graph, feed: Tensor) -> tuple[Tensor, ...]:
    """只取输出张量（按 `g.outputs` 顺序）。"""
    values = evaluate(g, feed)
    return tuple(values[o] for o in g.outputs)


def has_nan(t: Tensor) -> bool:
    return any(math.isnan(x) for row in t for x in row)


def eval_op(op: str, ins: Sequence[Tensor], attrs: dict[str, Any]) -> Tensor:
    """**单个算子**求值。常量折叠要用它 —— 它必须与 `evaluate` 走同一份实现，
    否则折叠出来的常量会与「不折叠时真跑一遍」的结果不一致，
    而这种不一致**不会报错**，只会让折叠后的图给出不同的数。
    """
    if op == "matmul":
        return _matmul(ins[0], ins[1])
    if op == "fused_mm_relu":
        mm = _matmul(ins[0], ins[1])
        return tuple(tuple(max(0.0, v) for v in row) for row in mm)
    if op == "fused_mm_bias_relu":
        mm = _matmul(ins[0], ins[1])
        bias = ins[2][0]
        return tuple(tuple(max(0.0, v + bias[j]) for j, v in enumerate(row)) for row in mm)
    if op in ("add", "mul"):
        return _broadcast_binop(ins[0], ins[1], op)
    if op == "relu":
        return tuple(tuple(max(0.0, v) for v in row) for row in ins[0])
    if op == "transpose":
        src = ins[0]
        return tuple(tuple(src[i][j] for i in range(len(src))) for j in range(len(src[0])))
    if op == "scale":
        s = float(attrs.get("s", 1.0))
        return tuple(tuple(v * s for v in row) for row in ins[0])
    raise GraphError(f"eval_op 不认识算子 {op}")
