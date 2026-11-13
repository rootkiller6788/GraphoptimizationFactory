"""核心对象：张量 / 节点 / 计算图 / 重写计划 / 试验。

三层身份要分清（`L4-rlfactory` 的坑 217）：
- **`Plan`** 是配置层的「一条重写序列」（一个 `tuple[str, ...]`，只有规则名）；
- **`Graph`** 是编译后的对象（真的能跑）；
- **`Trial`** 是跑完的结果对象（带目标向量与可行性）。

取属性之前先确认你在取谁。

张量一律用**不可变的嵌套 tuple**（`tuple[tuple[float, ...], ...]`）：可哈希 ⇒
可以直接做结构相等比较，CSE（公共子表达式消除）与图指纹都靠它。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Sequence

# --------------------------------------------------------------------------- #
# 张量
# --------------------------------------------------------------------------- #

Tensor = tuple[tuple[float, ...], ...]


def tensor(rows: Sequence[Sequence[float]]) -> Tensor:
    """行序列 → 不可变张量。**校验是矩形**，非矩形直接抛错（不静默补零）。"""
    out = tuple(tuple(float(x) for x in row) for row in rows)
    if not out:
        raise ValueError("张量不能为空")
    width = len(out[0])
    if width == 0:
        raise ValueError("张量不能有零宽")
    for r, row in enumerate(out):
        if len(row) != width:
            raise ValueError(f"张量不是矩形：第 0 行宽 {width}，第 {r} 行宽 {len(row)}")
    return out


def shape_of(t: Tensor) -> tuple[int, int]:
    return (len(t), len(t[0]))


def max_abs(t: Tensor) -> float:
    return max((abs(x) for row in t for x in row), default=0.0)


def zeros(rows: int, cols: int) -> Tensor:
    return tensor([[0.0] * cols for _ in range(rows)])


# --------------------------------------------------------------------------- #
# 节点与图
# --------------------------------------------------------------------------- #

#: 允许的算子。`fused_*` 是 `operator_fusion` 的产物 —— 它们的**数值语义**
#: 必须与未融合版本逐位相同（融合买到的是访存与 launch，不是算法）。
OPS = (
    "input",
    "const",
    "matmul",
    "add",
    "mul",
    "relu",
    "transpose",
    "fused_mm_relu",
    "fused_mm_bias_relu",
    "scale",
)


@dataclass(frozen=True)
class Node:
    """一个算子实例。`attrs` 是**排序后的**键值对（可哈希、可比较）。"""

    id: str
    op: str
    inputs: tuple[str, ...] = ()
    attrs: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.op not in OPS:
            raise ValueError(f"未知算子：{self.op}")
        object.__setattr__(self, "attrs", tuple(sorted(self.attrs)))

    def attr(self, key: str, default: Any = None) -> Any:
        for k, v in self.attrs:
            if k == key:
                return v
        return default

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "op": self.op, "inputs": list(self.inputs),
                "attrs": [[k, v] for k, v in self.attrs]}


class GraphError(Exception):
    """图不合法（缺父节点 / 有环 / 形状不匹配）。**一律抛错，不静默修补。**"""


@dataclass
class Graph:
    """有向无环计算图。`nodes` 的插入顺序**就是**一个拓扑序。

    不变量（每次 `add` 都校验）：
    - 节点的输入必须**已经存在**（于是构造顺序天然是拓扑序，不可能有环）；
    - `node_id` 不许重复；
    - `const` 节点必须在 `consts` 里有值，其余节点不许有。
    """

    nodes: dict[str, Node] = field(default_factory=dict)
    consts: dict[str, Tensor] = field(default_factory=dict)
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()

    # ---------------------------------------------------------------- 构造
    def add(self, op: str, ins: Sequence[str] = (), *,
            attrs: dict[str, Any] | None = None, const: Tensor | None = None,
            node_id: str | None = None) -> str:
        if node_id is None:
            node_id = f"n{len(self.nodes)}"
        if node_id in self.nodes:
            raise GraphError(f"节点已存在：{node_id}")
        for i in ins:
            if i not in self.nodes:
                raise GraphError(f"父节点不存在：{node_id} 需要 {i}")
        if op == "const":
            if const is None:
                raise GraphError(f"const 节点必须有值：{node_id}")
            self.consts[node_id] = const
        elif const is not None:
            raise GraphError(f"只有 const 节点能带值：{node_id}")
        self.nodes[node_id] = Node(node_id, op, tuple(ins),
                                   tuple(sorted((attrs or {}).items())))
        if op == "input":
            self.inputs = self.inputs + (node_id,)
        return node_id

    def set_outputs(self, outs: Sequence[str]) -> None:
        for o in outs:
            if o not in self.nodes:
                raise GraphError(f"输出节点不存在：{o}")
        self.outputs = tuple(outs)

    def copy(self) -> "Graph":
        return Graph(dict(self.nodes), dict(self.consts), self.inputs, self.outputs)

    # ---------------------------------------------------------------- 查询
    def topo(self) -> list[str]:
        """构造顺序即拓扑序（`add` 已强制父先于子）。"""
        return list(self.nodes)

    def consumers(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {nid: [] for nid in self.nodes}
        for nid, node in self.nodes.items():
            for i in node.inputs:
                out[i].append(nid)
        return out

    def ancestors(self, targets: Iterable[str]) -> set[str]:
        seen: set[str] = set()
        stack = list(targets)
        while stack:
            nid = stack.pop()
            if nid in seen:
                continue
            seen.add(nid)
            stack.extend(self.nodes[nid].inputs)
        return seen

    def op_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for node in self.nodes.values():
            counts[node.op] = counts.get(node.op, 0) + 1
        return counts

    # ------------------------------------------------------- 结构指纹 / 摘要
    def structural_key(self, node_id: str) -> str:
        """**与节点 id 无关**的结构指纹。

        两个子图算同一个东西 ⟺ 它们的 `structural_key` 相同 ——
        CSE 就是靠这条。`const` 节点的值进指纹（否则两个不同的常量会被当成同一个）。
        """
        cache: dict[str, str] = {}

        def walk(nid: str) -> str:
            if nid in cache:
                return cache[nid]
            node = self.nodes[nid]
            parts = [node.op]
            parts.extend(walk(i) for i in node.inputs)
            for k, v in node.attrs:
                parts.append(f"{k}={v!r}")
            if node.op == "const":
                parts.append(repr(self.consts[nid]))
            key = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:24]
            cache[nid] = key
            return key

        return walk(node_id)

    def digest(self) -> str:
        """整图指纹：**只吃输出与常量**，与节点 id、与创建顺序无关。

        两个结构等价但 id 不同的图必须给出同一个 digest —— 否则
        「这条重写有没有改变图」就没法用一次比较回答。
        """
        parts = sorted(self.structural_key(o) for o in self.outputs)
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]

    # ---------------------------------------------------------------- 序列化
    def to_dict(self) -> dict[str, Any]:
        """导出前**对节点 id 排序**：否则导出的顺序取决于「节点是何时被加进来的」，
        而 `from_dict` 重建时按拓扑序加 —— 同一张图导出两份 JSON（坑 29）。"""
        order = self.topo()
        return {
            "nodes": [self.nodes[n].to_dict() for n in order],
            "consts": {n: [list(r) for r in self.consts[n]] for n in order
                       if n in self.consts},
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "digest": self.digest(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Graph":
        g = cls()
        for nd in d["nodes"]:
            g.add(nd["op"], nd["inputs"], attrs={k: v for k, v in nd["attrs"]},
                  const=(tensor(d["consts"][nd["id"]]) if nd["id"] in d.get("consts", {})
                         else None),
                  node_id=nd["id"])
        g.set_outputs(d["outputs"])
        return g

    def to_dot(self, highlight: Iterable[str] = ()) -> str:
        hi = set(highlight)
        lines = ["digraph graphopt {", '  rankdir=LR;']
        for nid in self.topo():
            node = self.nodes[nid]
            label = node.op if not node.attrs else f"{node.op}\\n{node.attrs}"
            color = '"#c2410c"' if nid in hi else '"#1f2937"'
            lines.append(f'  "{nid}" [label="{label}", color={color}];')
        for nid in self.topo():
            for i in self.nodes[nid].inputs:
                lines.append(f'  "{i}" -> "{nid}";')
        lines.append("}")
        return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# 重写计划与试验
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Plan:
    """一条重写序列：**顺序是契约**。

    `[fusion, cse]` 与 `[cse, fusion]` 是两个不同的计划 —— 融合会改变 CSE
    能看见的东西，反过来也一样。这正是「候选是搜出来的」而不是「展开出来的」。
    """

    rules: tuple[str, ...]

    @property
    def id(self) -> str:
        return "p:" + hashlib.sha256("|".join(self.rules).encode("utf-8")).hexdigest()[:10]

    @property
    def depth(self) -> int:
        return len(self.rules)

    def label(self) -> str:
        return "[" + " → ".join(self.rules) + "]"

    def to_dict(self) -> dict[str, Any]:
        return {"plan_id": self.id, "rules": list(self.rules), "depth": self.depth}


@dataclass
class Trial:
    """一次评估的结果对象。**可行性与评分是两件事**（坑 238）。

    - `feasible`：语义漂移是否落在容差内（**硬约束**）；
    - `objectives`：全最大化口径的向量（可行时才参与前沿）；
    - `score`：仅供搜索器内部使用的标量化（`reject` 策略下不可行者给 `-1.0`
      而不是 `-inf` —— 见坑 257：`-inf` 会污染均值 / 标准差 / 归一化）。
    """

    plan: Plan
    searcher: str
    index: int
    feasible: bool
    drift: float
    objectives: tuple[float, ...]
    score: float
    applied: tuple[str, ...]
    nodes: int
    launches: int
    latency_us: float
    peak_bytes: int
    reject_reason: str = ""
    rank: int = -1
    on_frontier: bool = False
    dominated_by: int = 0

    @property
    def plan_id(self) -> str:
        return self.plan.id

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "rules": list(self.plan.rules),
            "searcher": self.searcher,
            "index": self.index,
            "feasible": self.feasible,
            "drift": self.drift,
            "objectives": [round(x, 12) for x in self.objectives],
            "score": round(self.score, 12),
            "applied": list(self.applied),
            "nodes": self.nodes,
            "launches": self.launches,
            "latency_us": round(self.latency_us, 9),
            "peak_bytes": self.peak_bytes,
            "reject_reason": self.reject_reason,
            "rank": self.rank,
            "on_frontier": self.on_frontier,
            "dominated_by": self.dominated_by,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Trial":
        """**显式按字段取值，绝不 `(**d)`** —— 多一个展示字段就会炸（坑 42）。"""
        return cls(
            plan=Plan(tuple(d["rules"])),
            searcher=d["searcher"],
            index=d["index"],
            feasible=d["feasible"],
            drift=d["drift"],
            objectives=tuple(d["objectives"]),
            score=d["score"],
            applied=tuple(d["applied"]),
            nodes=d["nodes"],
            launches=d["launches"],
            latency_us=d["latency_us"],
            peak_bytes=d["peak_bytes"],
            reject_reason=d["reject_reason"],
            rank=d["rank"],
            on_frontier=d["on_frontier"],
            dominated_by=d["dominated_by"],
        )


def iter_pairs(items: Sequence[Any]) -> Iterator[tuple[Any, Any]]:
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            yield items[i], items[j]
