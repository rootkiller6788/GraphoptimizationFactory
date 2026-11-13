"""重写规则库：8 条**合法**重写 + 3 条**刻意非法**的对照。

## 合法重写，按「是否逐位等价」分两族

| 规则 | 逐位等价？ | 漂移量级 |
| --- | --- | --- |
| `dead_node_elimination` | ✅ | 0 |
| `constant_folding` | ✅ | 0 |
| `common_subexpression` | ✅ | 0 |
| `double_transpose` | ✅ | 0 |
| `operator_fusion` | ✅ | 0（融合算子按**同一顺序**算同一批数） |
| `reassociate_add` | ❌ | ~1e-16（`(a+b)+c ≠ a+(b+c)`） |
| `matmul_reassociate` | ❌ | ~1e-15（结合律在浮点下不成立） |
| `scale_into_weight` | ❌ | ~1e-16（`(xW)·s ≠ x·(Ws)`） |

**这一栏是本项目存在的前提**：如果所有合法重写都逐位等价，容差就没有意义；
如果都不等价，「语义校验」就退化成「谁都不许动」。**缝在中间，缝有多宽是要测的。**

## 非法重写（只作对照，不进搜索空间）

`relu_to_identity` / `drop_bias` / `bias_epsilon(ε)`。作用是**证明校验器会报错而不是猜**
—— 一个「永远返回 True」的校验器与没有校验器在输出里长得一模一样（附录 C 261/275）。
`bias_epsilon` 的幅度可扫，用来**定位容差崖**。

## 三条踩出来的实现纪律

1. **规则自己用 `new.add(...)` 建节点并返回它的 id**，驱动器不预测 id。
   「预测下一个 id」的写法在驱动器改动时会**静默错位**，而错位的图照样能跑，
   只是跑的不是它该跑的东西。
2. **计划不许重叠。** 第一版 `reassociate_add` 逐节点独立决策，于是
   `o1` 被 `o5` 吸收、`o5` 又被 `out` 吸收，而 `out` 还要引用 `o1` 的新 id ⇒
   `父节点不存在：None`。修法是**沿左脊一次性规划整条链**（本文件 `_spine_plan`）。
3. **吸收的方向要写对。** 第一版 `drop_bias` 把 **matmul** 标成吸收、
   让 add 去别名它 —— 结果被吸收的 matmul 的 remap 是 `None`，
   于是它下游的 `relu` 复制失败。**丢 bias 要丢的是 add，不是 matmul。**
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable

from .evaluator import eval_op
from .graph import shape_table
from .schema import Graph, GraphError, Tensor, tensor

Shape = tuple[int, int]

#: 裁决：`None`（原样复制）/ `("drop",)` / `("id", 新图里的节点 id)`。
Verdict = None | tuple[str] | tuple[str, str]
#: `decide(nid, node, src, new, remap, hits)`；`hits` 是 `[0]`，**由规则自己决定什么算一次命中**。
Decider = Callable[..., Verdict]


def _bcast(a: Shape, b: Shape) -> Shape | None:
    """广播结果形状；不可广播返回 `None`（**不抛错** —— 调用方据此跳过这条规则）。"""
    if a == b:
        return a
    if a[1] == b[1] and (a[0] == 1 or b[0] == 1):
        return (max(a[0], b[0]), a[1])
    return None


def _rebuild(g: Graph, cfg: dict[str, Any], decide: Decider) -> tuple[Graph, int]:
    """按拓扑序重建图。

    **复制时若某个输入没有映射，直接抛 `GraphError`** —— 不静默级联删除。
    静默级联会把「融合把中间节点吸收了」变成「把整条支路删了」，
    而两者在读数上都只是「节点变少了」。
    """
    new = Graph()
    remap: dict[str, str | None] = {}
    hits = [0]
    for nid in g.topo():
        node = g.nodes[nid]
        v = decide(nid, node, g, new, remap, hits)
        if v is None:
            missing = [i for i in node.inputs if remap.get(i) is None]
            if missing:
                raise GraphError(f"{nid}({node.op}) 复制失败：输入 {missing} 已被吸收/删除")
            ins = tuple(remap[i] for i in node.inputs)          # type: ignore[misc]
            remap[nid] = new.add(node.op, ins, attrs=dict(node.attrs),
                                 const=g.consts.get(nid))
        elif v[0] == "drop":
            remap[nid] = None
        else:
            remap[nid] = v[1]
    outs = [remap[o] for o in g.outputs]
    if any(o is None for o in outs):
        raise GraphError(f"重写把输出节点删掉了：{g.outputs} → {outs}")
    new.set_outputs(outs)                                       # type: ignore[arg-type]
    shape_table(new, cfg)          # 重建即校验：形状错了立刻炸，不留到求值时
    return new, hits[0]


def _shapes(g: Graph, cfg: dict[str, Any]) -> dict[str, Shape]:
    return shape_table(g, cfg)


# --------------------------------------------------------------------------- #
# 合法规则
# --------------------------------------------------------------------------- #

def dead_node_elimination(g: Graph, cfg: dict[str, Any]) -> tuple[Graph, int]:
    """删掉任何**不接到输出**的节点。逐位等价（少算的东西本来就不影响输出）。"""
    live = g.ancestors(g.outputs)

    def decide(nid, node, src, new, remap, hits):
        if nid in live:
            return None
        hits[0] += 1
        return ("drop",)

    return _rebuild(g, cfg, decide)


def constant_folding(g: Graph, cfg: dict[str, Any]) -> tuple[Graph, int]:
    """把「输入全是常量」的节点直接算成一个常量。**一遍内级联**。

    用的是 `evaluator.eval_op` —— 与真跑一遍**同一份实现**。两份实现漂移的后果
    是「折叠后的图给出不同的数」，而且不会报错（附录 C 53 的同一条纪律）。
    """
    def decide(nid, node, src, new, remap, hits):
        if node.op in ("input", "const"):
            return None
        ins = tuple(remap[i] for i in node.inputs)              # type: ignore[misc]
        if not all(new.nodes[i].op == "const" for i in ins):
            return None
        folded = eval_op(node.op, [new.consts[i] for i in ins], dict(node.attrs))
        hits[0] += 1
        return ("id", new.add("const", (), const=folded))

    return _rebuild(g, cfg, decide)


def common_subexpression(g: Graph, cfg: dict[str, Any]) -> tuple[Graph, int]:
    """结构相同的子表达式只算一遍。

    判据是**新图里的**结构键（含常量值），不是节点 id —— 用 id 判等于没判。
    级联是自动的：`u2 → u1` 之后，`relu(u2)` 与 `relu(u1)` 的输入键也相同，一遍内合并。

    ⚠️ **命中数只数真正的别名**。第一版把「新建了一个节点」也算成命中，
    于是 49 个节点报出 49 次命中 —— 看起来像「每个节点都被合并了一次」，
    实际只合并了 2 个。**计数口径错了，读数就全错。**
    """
    by_key: dict[str, str] = {}

    def key_of(op: str, ins: tuple[str, ...], attrs, const, new: Graph) -> str:
        parts = [op]
        parts.extend(new.structural_key(i) for i in ins)
        parts.extend(f"{k}={v!r}" for k, v in attrs)
        if const is not None:
            parts.append(repr(const))
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()

    def decide(nid, node, src, new, remap, hits):
        ins = tuple(remap[i] for i in node.inputs)              # type: ignore[misc]
        const = src.consts.get(nid)
        key = key_of(node.op, ins, node.attrs, const, new)
        if node.op != "input" and key in by_key:
            hits[0] += 1
            return ("id", by_key[key])
        created = new.add(node.op, ins, attrs=dict(node.attrs), const=const)
        by_key[key] = created
        return ("id", created)

    return _rebuild(g, cfg, decide)


def double_transpose(g: Graph, cfg: dict[str, Any]) -> tuple[Graph, int]:
    """`transpose(transpose(x)) → x`。逐位等价（转置是纯置换）。

    ⚠️ 别名要指向**祖父**（`inner.inputs[0]`），不是父节点 ——
    指向父节点得到的是 `transpose(x)`，图仍然合法、仍然能跑，
    只是「消掉了一重转置」而**形状悄悄变了**（第一版就是这么错的，
    直到下游 `add` 报「形状不可广播 (8,4) vs (1,8)」才现形）。
    """
    def decide(nid, node, src, new, remap, hits):
        if node.op != "transpose":
            return None
        inner = src.nodes[node.inputs[0]]
        if inner.op != "transpose":
            return None
        hits[0] += 1
        return ("id", remap[inner.inputs[0]])                   # type: ignore[return-value]

    return _rebuild(g, cfg, decide)


def operator_fusion(g: Graph, cfg: dict[str, Any]) -> tuple[Graph, int]:
    """`relu(add(matmul(a,b), c)) → fused_mm_bias_relu(a,b,c)`（以及无 bias 的形式）。

    **逐位等价**：融合算子在 `evaluator` 里按**同一顺序**算 `matmul → add → relu`，
    所以融合买到的是**访存与 launch**，不是算法。这条必须成立，否则
    「融合不改变语义」就成了口号 —— `acceptance` 里有一条用例专门钉它。

    两个「恰好一个消费者」的条件不能省：不满足时融合会**复制**计算
    （结果一样但更慢），而那种退化在读数上看起来像「融合没用」。
    """
    shapes = _shapes(g, cfg)
    cons = g.consumers()
    fuse: dict[str, tuple[str, ...]] = {}
    absorbed: set[str] = set()

    for nid, node in g.nodes.items():
        if node.op != "relu" or nid in absorbed:
            continue
        src_id = node.inputs[0]
        snode = g.nodes[src_id]
        if snode.op == "matmul" and len(cons[src_id]) == 1:
            fuse[nid] = ("plain",) + snode.inputs
            absorbed.add(src_id)
        elif snode.op == "add" and len(cons[src_id]) == 1:
            mms = [i for i in snode.inputs if g.nodes[i].op == "matmul"]
            if len(mms) != 1 or len(cons[mms[0]]) != 1 or mms[0] in absorbed:
                continue
            other = snode.inputs[1] if snode.inputs[0] == mms[0] else snode.inputs[0]
            if shapes[other][0] != 1 or other in absorbed:
                continue                                   # 只接受逐通道 bias，且未被吸收
            a, b = g.nodes[mms[0]].inputs
            if a in absorbed or b in absorbed:
                continue
            fuse[nid] = ("bias", a, b, other)
            absorbed.add(src_id)
            absorbed.add(mms[0])

    def decide(nid, node, src, new, remap, hits):
        if nid in absorbed and nid not in fuse:
            return ("drop",)                # 被吸收，不计命中（真正的命中记在 relu 上）
        spec = fuse.get(nid)
        if spec is None:
            return None
        hits[0] += 1
        if spec[0] == "plain":
            return ("id", new.add("fused_mm_relu",
                                  (remap[spec[1]], remap[spec[2]])))  # type: ignore[arg-type]
        return ("id", new.add("fused_mm_bias_relu",
                              (remap[spec[1]], remap[spec[2]], remap[spec[3]])))  # type: ignore[arg-type]

    return _rebuild(g, cfg, decide)


def _mm_shape(d: Shape, r: Shape) -> Shape | None:
    """矩阵乘的形状规则：内维必须相等。**不能复用 `_bcast`** ——
    广播是逐元素的规则，用在矩阵乘上会把 `matmul(h, w)` 判成不合法
    （`(4,8)` 与 `(8,8)` 广播不了），于是整条规则静默命中 0 次。"""
    if d[1] != r[0]:
        return None
    return (d[0], r[1])


def _spine_plan(g: Graph, cfg: dict[str, Any], op: str,
                combine: Callable[[Shape, Shape], Shape | None]
                ) -> tuple[dict[str, tuple[str, str, list[str]]], set[str]]:
    """沿**左脊**一次性规划整条链，避免计划重叠。

    返回 `(plan, claimed)`：`plan[顶层节点] = (a, b, rights)`，其中
    `rights` 从外到内（`[d_1, d_2, …]`）；`claimed` 是除顶层外**全部脊节点**。

    表达式 `((…((a ⊕ b) ⊕ d_{k-1}) …) ⊕ d_1)` 重写成右倾形式
    `a ⊕ (b ⊕ (d_{k-1} ⊕ (… ⊕ d_1)))`。**逐节点独立决策会重叠**，
    而重叠的表现是 `父节点不存在：None`（见模块文档字符串第 2 条）。
    """
    shapes = _shapes(g, cfg)
    cons = g.consumers()
    plan: dict[str, tuple[str, str, list[str]]] = {}
    claimed: set[str] = set()

    # **从最外层往内**规划：否则内层链先占坑，外层链会在「基座已被占」处停下，
    # 得到一个「基座其实会被丢掉」的计划 ⇒ `remap[a] is None`。
    # 反序还能让更长的链赢（覆盖更好），而且顺序是确定的。
    for nid in reversed(g.topo()):
        node = g.nodes[nid]
        if node.op != op or nid in claimed:
            continue
        # 先走完脊：`chain = [X_1(顶), X_2, …, X_k(基座)]`
        chain = [nid]
        cur = nid
        while True:
            left = g.nodes[cur].inputs[0]
            if (g.nodes[left].op == op and cons[left] == [cur]
                    and left not in claimed):
                chain.append(left)
                cur = left
            else:
                break
        if len(chain) < 2 or chain[-1] in claimed:
            continue
        # `rights` **只收脊上非基座节点的右操作数**（k−1 个）。
        # 第一版把基座自己的右操作数也收了进来 ⇒ `b` 在重建时出现两次，
        # 而 `matmul(w1, matmul(w1, w2))` 形状完全合法、照样能跑，只是算的是另一个式子。
        a, b = g.nodes[chain[-1]].inputs
        rights = [g.nodes[x].inputs[1] for x in chain[:-1]]
        # **存活检查**：重建时要引用的每个节点都必须不在 `claimed` 里。
        # 少了这一条，计划就会引用一个「会被别的计划丢掉」的节点。
        if a in claimed or b in claimed or any(r in claimed for r in rights):
            continue
        # 形状预演：右倾重建的每一步都要可组合，且最终形状与原来一致
        cur_shape = shapes[rights[0]]
        ok = True
        for d in rights[1:] + [b, a]:
            nxt = combine(shapes[d], cur_shape)
            if nxt is None:
                ok = False
                break
            cur_shape = nxt
        if not ok or cur_shape != shapes[nid]:
            continue
        plan[nid] = (a, b, rights)
        claimed.update(chain[1:])
    return plan, claimed


def _apply_spine(new: Graph, remap: dict[str, str | None], binop: str,
                 a: str, b: str, rights: list[str]) -> str:
    """按右倾形式建节点，返回顶层新节点 id。"""
    r = remap[rights[0]]
    for d in rights[1:]:
        r = new.add(binop, (remap[d], r))                       # type: ignore[arg-type]
    r = new.add(binop, (remap[b], r))                           # type: ignore[arg-type]
    return new.add(binop, (remap[a], r))                        # type: ignore[arg-type]


def reassociate_add(g: Graph, cfg: dict[str, Any]) -> tuple[Graph, int]:
    """`add(add(a,b), c) → add(a, add(b,c))`。**浮点下不逐位等价**（这就是它的用途）。"""
    plan, claimed = _spine_plan(g, cfg, "add", _bcast)

    def decide(nid, node, src, new, remap, hits):
        if nid in claimed:
            return ("drop",)
        spec = plan.get(nid)
        if spec is None:
            return None
        hits[0] += 1
        a, b, rights = spec
        return ("id", _apply_spine(new, remap, "add", a, b, rights))

    return _rebuild(g, cfg, decide)


def matmul_reassociate(g: Graph, cfg: dict[str, Any]) -> tuple[Graph, int]:
    """`matmul(matmul(a,b), c) → matmul(a, matmul(b,c))`。**浮点下不逐位等价**。

    与 `reassociate_add` 共用脊规划 —— 两者的区别只在算子名，
    而「两处各写一遍脊规划」迟早会漂移成两套语义。
    """
    plan, claimed = _spine_plan(g, cfg, "matmul", _mm_shape)

    def decide(nid, node, src, new, remap, hits):
        if nid in claimed:
            return ("drop",)
        spec = plan.get(nid)
        if spec is None:
            return None
        hits[0] += 1
        a, b, rights = spec
        return ("id", _apply_spine(new, remap, "matmul", a, b, rights))

    return _rebuild(g, cfg, decide)


def scale_into_weight(g: Graph, cfg: dict[str, Any]) -> tuple[Graph, int]:
    """`scale(matmul(x, W), s) → matmul(x, W·s)`（`W` 必须是常量）。**不逐位等价**。"""
    cons = g.consumers()
    plan: dict[str, tuple[str, str, float]] = {}
    absorbed: set[str] = set()

    for nid, node in g.nodes.items():
        if node.op != "scale":
            continue
        left = node.inputs[0]
        if g.nodes[left].op != "matmul" or len(cons[left]) != 1 or left in absorbed:
            continue
        x, w = g.nodes[left].inputs
        if w not in g.consts:
            continue
        plan[nid] = (x, w, float(node.attr("s", 1.0)))
        absorbed.add(left)

    def decide(nid, node, src, new, remap, hits):
        if nid in absorbed:
            return ("drop",)
        spec = plan.get(nid)
        if spec is None:
            return None
        hits[0] += 1
        x, w, s = spec
        scaled = tensor([[v * s for v in row] for row in src.consts[w]])
        new_w = new.add("const", const=scaled)
        return ("id", new.add("matmul", (remap[x], new_w)))     # type: ignore[arg-type]

    return _rebuild(g, cfg, decide)


# --------------------------------------------------------------------------- #
# 非法规则（只作对照）
# --------------------------------------------------------------------------- #

def relu_to_identity(g: Graph, cfg: dict[str, Any]) -> tuple[Graph, int]:
    """把 `relu` 直接换成它的输入。**非法** —— 负值不再被截断。"""
    def decide(nid, node, src, new, remap, hits):
        if node.op != "relu":
            return None
        hits[0] += 1
        return ("id", remap[node.inputs[0]])                    # type: ignore[return-value]

    return _rebuild(g, cfg, decide)


def drop_bias(g: Graph, cfg: dict[str, Any]) -> tuple[Graph, int]:
    """丢掉 bias 项。**非法** —— 输出整体偏移。

    ⚠️ `add(matmul(x,W), b) → matmul(x,W)`：被替换的是 **add**，
    matmul 必须**留着**。第一版把 matmul 标成吸收、让 add 去别名它，
    于是被吸收的 matmul 的 remap 是 `None`，下游 `relu` 复制直接失败。
    """
    plan: dict[str, tuple[str, ...]] = {}

    for nid, node in g.nodes.items():
        if node.op == "fused_mm_bias_relu":
            plan[nid] = ("fused", node.inputs[0], node.inputs[1])
        elif node.op == "add":
            mms = [i for i in node.inputs if g.nodes[i].op == "matmul"]
            if len(mms) == 1:
                plan[nid] = ("alias", mms[0])

    def decide(nid, node, src, new, remap, hits):
        spec = plan.get(nid)
        if spec is None:
            return None
        hits[0] += 1
        if spec[0] == "fused":
            return ("id", new.add("fused_mm_relu",
                                  (remap[spec[1]], remap[spec[2]])))  # type: ignore[arg-type]
        return ("id", remap[spec[1]])                           # type: ignore[return-value]

    return _rebuild(g, cfg, decide)


def bias_epsilon(g: Graph, cfg: dict[str, Any], eps: float) -> tuple[Graph, int]:
    """把所有 1×C 的常量按 `w + ε·max|w|` 扰动。**非法**。

    `ε` 可扫 —— 它是**定位容差崖的那把尺子**：合法重写的漂移在 1e-16 量级，
    这条规则可以把漂移放到 1e-12 ~ 1e-1 之间的任意位置，于是
    「容差多大才会开始收下错的答案」变成一个**可测量**的数。
    """
    cons = g.consumers()

    def decide(nid, node, src, new, remap, hits):
        if node.op != "const":
            return None
        wt = src.consts[nid]
        if len(wt) != 1 or not any(src.nodes[c].op == "add" for c in cons[nid]):
            return None
        m = max(abs(v) for v in wt[0]) or 1.0
        hits[0] += 1
        return ("id", new.add("const", (), const=tensor([[v + eps * m for v in wt[0]]])))

    return _rebuild(g, cfg, decide)


# --------------------------------------------------------------------------- #
# 注册表
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Rule:
    name: str
    kind: str            # "legal" | "illegal"
    note: str
    fn: Callable[[Graph, dict[str, Any]], tuple[Graph, int]]


LEGAL_RULES: tuple[Rule, ...] = (
    Rule("dead_node_elimination", "legal",
         "删掉不接到输出的节点（逐位等价）", dead_node_elimination),
    Rule("constant_folding", "legal",
         "把全常量子图算成常量（逐位等价）", constant_folding),
    Rule("common_subexpression", "legal",
         "结构相同的子表达式只算一遍（逐位等价）", common_subexpression),
    Rule("double_transpose", "legal",
         "transpose(transpose(x)) → x（逐位等价）", double_transpose),
    Rule("operator_fusion", "legal",
         "matmul→add→relu 融合（逐位等价，省访存与 launch）", operator_fusion),
    Rule("reassociate_add", "legal",
         "(a+b)+c → a+(b+c)（**不逐位等价**，漂移 ~1e-16）", reassociate_add),
    Rule("matmul_reassociate", "legal",
         "(a@b)@c → a@(b@c)（**不逐位等价**，漂移 ~1e-15）", matmul_reassociate),
    Rule("scale_into_weight", "legal",
         "scale(matmul(x,W),s) → matmul(x,W·s)（**不逐位等价**，漂移 ~1e-16）",
         scale_into_weight),
)

CONTROL_RULES: tuple[Rule, ...] = (
    Rule("relu_to_identity", "illegal", "relu 换成恒等（负值不再截断）", relu_to_identity),
    Rule("drop_bias", "illegal", "丢掉 bias 项（输出整体偏移）", drop_bias),
)

RULES: dict[str, Rule] = {r.name: r for r in LEGAL_RULES + CONTROL_RULES}
LEGAL_NAMES: tuple[str, ...] = tuple(r.name for r in LEGAL_RULES)
CONTROL_NAMES: tuple[str, ...] = tuple(r.name for r in CONTROL_RULES)


def apply_rule(g: Graph, name: str, cfg: dict[str, Any]) -> tuple[Graph, int]:
    """施加一条规则。**只施加一次**（不迭代到不动点）—— 迭代会掩盖顺序效应，
    而顺序正是本项目要搜的东西。

    规则在某张图上没有匹配点时会**原样返回、命中 0 次**（那是正常情形）。
    **不适用要在规则内部判断**（形状预演），不在这里兜 ——

    ⚠️ 第一版在这里写了 `except GraphError: return g.copy(), 0`，后果是
    `reassociate_add` 的「计划重叠 ⇒ 父节点不存在 None」这个**真 bug**
    被伪装成「这条规则在这张图上不适用」，读数是 0 命中，看起来完全正常。
    **一个吞掉异常的兜底，等于一条静默失效的通道。**
    """
    if name not in RULES:
        raise GraphError(f"未知规则：{name}")
    return RULES[name].fn(g, cfg)
