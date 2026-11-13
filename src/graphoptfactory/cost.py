"""解析式成本模型 + 活跃期（liveness）峰值内存。

**这一层全部是「算出来的」，没有一维是测出来的。** 把假设标成测量是这类报告
最危险的失真（附录 C 276）—— 所以每个量都带一个 `source` 标签，
产物里也会原样落盘，读者不必猜。

常数全在 `configs/default.json` 的 `cost` 段：

| 常数 | 含义 | 这一维的**假设** |
| --- | --- | --- |
| `launch_overhead_us` | 每次 kernel launch 的固定开销 | 与算子规模无关 |
| `flops_per_us` | 吞吐 | 所有算子同一个吞吐（真实机器上不成立） |
| `bytes_per_us` | 带宽 | 访存与计算不重叠 |
| `bytes_per_elem` | 单元素字节数 | fp32 |

**「融合省时间」这个结论只在「launch 开销不可忽略」这个假设下成立** ——
把 `launch_overhead_us` 扫到 0，融合的收益就只剩访存那一份。这条要写进报告，
否则读者会把一个受控演示当成真机结论。
"""

from __future__ import annotations

from typing import Any

from .schema import Graph, Tensor

#: 每个成本量的**来源标签**。`analytic` = 公式算出来的，`structural` = 数出来的。
COST_SOURCE: dict[str, str] = {
    "nodes": "structural",
    "launches": "structural",
    "const_nodes": "structural",
    "latency_us": "analytic",
    "peak_bytes": "analytic",
    "flops": "analytic",
    "bytes_moved": "analytic",
}


def _elems(shape: tuple[int, int]) -> int:
    return shape[0] * shape[1]


def _flops(op: str, ins: list[tuple[int, int]], out: tuple[int, int]) -> int:
    if op in ("matmul", "fused_mm_relu", "fused_mm_bias_relu"):
        m, k = ins[0]
        n = ins[1][1]
        base = 2 * m * n * k
        if op == "fused_mm_bias_relu":
            base += 2 * _elems(out)      # 逐元素加 + relu
        elif op == "fused_mm_relu":
            base += _elems(out)
        return base
    return _elems(out)


def _bytes_moved(op: str, ins: list[tuple[int, int]], out: tuple[int, int], bpe: int) -> int:
    """访存量 = 读全部输入 + 写输出。**融合买到的主要就是这一项**。"""
    return (sum(_elems(s) for s in ins) + _elems(out)) * bpe


def launches(g: Graph) -> int:
    """kernel launch 次数 = 非 `input` / 非 `const` 的节点数。

    `const` 不是一次 launch（它是权重，随图常驻），`input` 也不是。
    这个口径要写死在一处，否则「融合省了多少 launch」在两处会给出两个数。
    """
    return sum(1 for n in g.nodes.values() if n.op not in ("input", "const"))


def peak_bytes(g: Graph, shapes: dict[str, tuple[int, int]], bpe: int) -> int:
    """活跃期分析的峰值内存。

    一个张量从**被产生**活到**最后一次被读**；图输出活到最后。
    输入与常量从第 0 步就活着（它们在图外产生）。

    ⚠️ 这个量**对「节点顺序」敏感**：同一张图换个拓扑序会得到不同的峰值。
    本项目一律用构造顺序，所以它是确定的 —— 但换实现时这里会静默改变。
    """
    order = g.topo()
    pos = {nid: i for i, nid in enumerate(order)}
    cons = g.consumers()
    last_use: dict[str, int] = {}
    for nid in order:
        uses = [pos[c] for c in cons[nid]]
        last_use[nid] = max(uses) if uses else pos[nid]
    for o in g.outputs:
        last_use[o] = len(order) - 1

    live: dict[str, int] = {}
    peak = 0
    for i, nid in enumerate(order):
        live[nid] = _elems(shapes[nid]) * bpe
        peak = max(peak, sum(b for k, b in live.items() if last_use[k] >= i))
    return peak


def total_flops(g: Graph, shapes: dict[str, tuple[int, int]]) -> int:
    return sum(_flops(n.op, [shapes[i] for i in n.inputs], shapes[nid])
               for nid, n in g.nodes.items() if n.op not in ("input", "const"))


def metrics(g: Graph, shapes: dict[str, tuple[int, int]], cfg: dict[str, Any]) -> dict[str, Any]:
    """一次算出全部成本读数。返回的每个键都在 `COST_SOURCE` 里有来源标签。"""
    c = cfg["cost"]
    bpe = int(c["bytes_per_elem"])
    overhead = float(c["launch_overhead_us"])
    fpu = float(c["flops_per_us"])
    bpu = float(c["bytes_per_us"])

    lat = 0.0
    fl = 0
    moved_total = 0
    for nid, n in g.nodes.items():
        if n.op in ("input", "const"):
            continue
        ins = [shapes[i] for i in n.inputs]
        f = _flops(n.op, ins, shapes[nid])
        moved = _bytes_moved(n.op, ins, shapes[nid], bpe)
        fl += f
        moved_total += moved
        lat += overhead + f / fpu + moved / bpu

    return {
        "nodes": len(g.nodes),
        "launches": launches(g),
        "flops": fl,
        "bytes_moved": moved_total,
        "latency_us": lat,
        "peak_bytes": peak_bytes(g, shapes, bpe),
        "const_nodes": sum(1 for n in g.nodes.values() if n.op == "const"),
    }


def sweep_launch_overhead(cfg: dict[str, Any], values: list[float],
                          baseline: dict[str, Any], best: dict[str, Any]) -> list[dict[str, Any]]:
    """扫 `launch_overhead_us`，看「融合的收益」有多少来自 launch 开销。

    一条**软假设**的系数扫描（附录 C 244 的同一条纪律）：凡「某机制买到 X」
    的结论，都要报告它在哪个系数区间才成立。

    除 `launch_overhead_us` 外全部常数保持默认 —— 一次只动一个变量。
    """
    c = cfg["cost"]
    fpu = float(c["flops_per_us"])
    bpu = float(c["bytes_per_us"])

    def lat(m: dict[str, Any], ov: float) -> float:
        return ov * m["launches"] + m["flops"] / fpu + m["bytes_moved"] / bpu

    out: list[dict[str, Any]] = []
    for ov in values:
        b = lat(baseline, ov)
        t = lat(best, ov)
        out.append({
            "launch_overhead_us": ov,
            "baseline_latency_us": round(b, 9),
            "best_latency_us": round(t, 9),
            "speedup": round(b / t, 9) if t else 0.0,
        })
    return out
