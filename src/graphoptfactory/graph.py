"""形状推导与基线计算图构建。

基线图是**刻意构造**的：它必须让**每一条重写规则**都有靶子，否则端到端跑出来
什么也演示不了（同 `L0` 的坑 6：demo 数据要让每条规则都被触发一次）。

埋进去的靶子：

| 规则 | 靶子 |
| --- | --- |
| `operator_fusion` | 主干每层的 `matmul → add → relu` 链 |
| `common_subexpression` | 两条支路各自重建了一遍 `matmul(x, W_s)` / `relu(...)` |
| `constant_folding` | 一条**完全由常量构成**的支路 |
| `dead_node_elimination` | 一条不接到任何输出的支路 |
| `double_transpose` | `transpose(transpose(h))` |
| `reassociate_add` | `((t + b0) + b1)` |
| `matmul_reassociate` | `matmul(matmul(h, W1), W2)` |
| `scale_into_weight` | `scale(matmul(h, W3), s)` |

**种子成分表就是「什么参数影响什么产物」的声明**（附录 C 239）：权重的根种子
只用 4 个**任务参数**（层数 / 行数 / 维度 / 权重幅度），
`search.budget` 这类只影响下游代价的参数**不许**进上游种子 ——
否则改一个不相干的参数会连带改掉整张图，而「两次 run 结果不同」会被归因错。
"""

from __future__ import annotations

from typing import Any

from .rng import Rng, derive_seed
from .schema import Graph, GraphError, Tensor, tensor

#: 规则名 → 靶子描述（报告与 `rules` 命令共用，避免两处写死）。
TARGET_NOTES: dict[str, str] = {
    "operator_fusion": "主干每层的 matmul→add→relu",
    "common_subexpression": "两条支路各建一遍 matmul(x,W_s)",
    "constant_folding": "一条全常量支路",
    "dead_node_elimination": "一条不接输出的支路",
    "double_transpose": "transpose(transpose(h))",
    "reassociate_add": "((t+b0)+b1)",
    "matmul_reassociate": "matmul(matmul(h,W1),W2)",
    "scale_into_weight": "scale(matmul(h,W3), s)",
}


def _rand_matrix(rng: Rng, rows: int, cols: int, scale: float) -> Tensor:
    return tensor([[rng.uniform(-scale, scale) for _ in range(cols)]
                   for _ in range(rows)])


def shape_table(g: Graph, cfg: dict[str, Any]) -> dict[str, tuple[int, int]]:
    """逐节点形状推导。**遇到不匹配直接抛 `GraphError`**（不静默取其一）。

    这是重写安全的第一道闸：融合 / 重结合都要求形状兼容，而形状错误
    在没有推导器的时候只会表现为「跑出来的数不对」。
    """
    rows = int(cfg["graph"]["rows"])
    dim = int(cfg["graph"]["dim"])
    out: dict[str, tuple[int, int]] = {}

    def sh(nid: str) -> tuple[int, int]:
        if nid in out:
            return out[nid]
        node = g.nodes[nid]
        ins = [sh(i) for i in node.inputs]
        op = node.op
        if op == "input":
            res = (rows, dim)
        elif op == "const":
            res = (len(g.consts[nid]), len(g.consts[nid][0]))
        elif op in ("matmul", "fused_mm_relu", "fused_mm_bias_relu"):
            a, b = ins[0], ins[1]
            if a[1] != b[0]:
                raise GraphError(f"{nid}: {op} 内维不匹配 {a} × {b}")
            res = (a[0], b[1])
        elif op in ("add", "mul"):
            a, b = ins[0], ins[1]
            if a == b:
                res = a
            elif a[1] == b[1] and (a[0] == 1 or b[0] == 1):
                res = (max(a[0], b[0]), a[1])
            else:
                raise GraphError(f"{nid}: {op} 形状不可广播 {a} vs {b}")
        elif op == "relu":
            res = ins[0]
        elif op == "transpose":
            res = (ins[0][1], ins[0][0])
        elif op == "scale":
            res = ins[0]
        else:  # pragma: no cover - OPS 已封口
            raise GraphError(f"{nid}: 未知算子 {op}")
        out[nid] = res
        return res

    for nid in g.topo():
        sh(nid)
    return out


def build_baseline(cfg: dict[str, Any]) -> Graph:
    """构造基线图。**纯函数**：同一份 `cfg` ⇒ 逐位相同的图。"""
    gc = cfg["graph"]
    layers = int(gc["n_layers"])
    rows = int(gc["rows"])
    dim = int(gc["dim"])
    scale = float(gc["weight_scale"])

    # 根种子只用任务参数 —— 见模块文档字符串。
    rng = Rng(derive_seed(cfg["seed"], "weights", layers, rows, dim, scale))
    g = Graph()

    def wmat(rows_: int, cols_: int) -> str:
        return g.add("const", const=_rand_matrix(rng, rows_, cols_, scale))

    def wvec(cols_: int) -> str:
        return g.add("const", const=_rand_matrix(rng, 1, cols_, scale))

    x = g.add("input")

    # ---- 主干：matmul → add(bias) → relu（operator_fusion 的靶子） ----
    h = x
    for _ in range(layers):
        w = wmat(dim, dim)
        b = wvec(dim)
        a = g.add("matmul", (h, w))
        a = g.add("add", (a, b))
        h = g.add("relu", (a,))

    # ---- 公共子表达式：同一条子表达式被**重建**了一遍 ----
    w_shared = wmat(dim, dim)
    u1 = g.add("matmul", (x, w_shared))
    u2 = g.add("matmul", (x, w_shared))          # 与 u1 结构完全相同
    p = g.add("relu", (u1,))
    q = g.add("relu", (u2,))

    # ---- 常量折叠：整条支路只由常量构成（c1 取 rows×dim 以对齐形状） ----
    c1 = wmat(rows, dim)
    c2 = wmat(dim, dim)
    c3 = wvec(dim)
    cf = g.add("matmul", (c1, c2))
    cf = g.add("add", (cf, c3))

    # ---- 死节点：不接到任何输出 ----
    w_dead = wmat(dim, dim)
    d1 = g.add("matmul", (h, w_dead))
    g.add("relu", (d1,))                          # 故意不接输出

    # ---- 双转置 ----
    t1 = g.add("transpose", (h,))
    t2 = g.add("transpose", (t1,))                # transpose(transpose(h)) ⇒ h

    # ---- 加法重结合：((t2 + b0) + b1) ----
    b0 = wvec(dim)
    b1 = wvec(dim)
    k1 = g.add("add", (t2, b0))
    k2 = g.add("add", (k1, b1))

    # ---- matmul 重结合：matmul(matmul(h, W1), W2) ----
    w1 = wmat(dim, dim)
    w2 = wmat(dim, dim)
    m1 = g.add("matmul", (h, w1))
    m2 = g.add("matmul", (m1, w2))

    # ---- scale 进权重 ----
    w3 = wmat(dim, dim)
    sc_mm = g.add("matmul", (h, w3))
    sc = g.add("scale", (sc_mm,), attrs={"s": 1.5})

    # ---- 收拢到单一输出 ----
    o1 = g.add("add", (h, p))
    o2 = g.add("add", (q, cf))
    o3 = g.add("add", (t2, k2))
    o4 = g.add("add", (m2, sc))
    o5 = g.add("add", (o1, o2))
    o6 = g.add("add", (o3, o4))
    out = g.add("add", (o5, o6))
    g.set_outputs((out,))

    shape_table(g, cfg)      # 构造即校验
    return g


def build_probes(cfg: dict[str, Any]) -> list[Tensor]:
    """探针输入。**只吃 `probes` 段与任务参数**，与搜索预算无关。

    `semantics.drift` 是在这些探针上量的 —— 探针换了，「语义漂移」这个读数就换了，
    所以它必须只由**它自己的**参数决定（附录 C 239 的同一条纪律）。

    返回的是**张量列表**（按 `graph.inputs` 的顺序喂给图），不是伪造的键 ——
    伪造键会让「图有几个输入」这件事与探针的构造耦合在一起。
    """
    pc = cfg["probes"]
    n = int(pc["n_probe"])
    rows = int(cfg["graph"]["rows"])
    dim = int(cfg["graph"]["dim"])
    scale = float(pc["scale"])
    rng = Rng(derive_seed(cfg["seed"], "probes", n, rows, dim, scale))
    return [_rand_matrix(rng, rows, dim, scale) for _ in range(n)]
