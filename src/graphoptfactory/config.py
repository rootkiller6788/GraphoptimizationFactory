"""配置：`DEFAULTS` + 深合并 + 加载 + 校验 + 协议指纹。

**所有阈值 / 权重 / 容差都在这里，代码里不硬编码**（工程约定）。
`configs/default.json` 是 `DEFAULTS` 的落盘副本，`dump-config` 会重新生成它。

⚠️ 深合并里**列表是整体替换、磁盘优先** —— 于是「改了代码里的声明、重跑，
数字一字未变」是一个真实踩过的坑（附录 C 204）。所以有一条保险丝
`no_stale_config_keys`：磁盘上多出来的键会让流水线报错，而不是被静默忽略。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .rng import digest_of
from .rewrites import LEGAL_NAMES

DEFAULTS: dict[str, Any] = {
    "seed": 20261003,

    # ---- 任务参数：**根种子只用这里的值**（附录 C 239） ----
    "graph": {
        "n_layers": 3,
        "rows": 4,
        "dim": 8,
        "weight_scale": 0.5,
    },
    "probes": {
        "n_probe": 3,
        "scale": 0.7,
    },

    # ---- 硬约束：语义容差 ----
    "semantics": {
        #: **「未测量时会选的容差」**，不是推荐值。
        #: 本项目的结论之一就是它取错了：合法重写的漂移在 1.5e-16 量级，
        #: 而 1e-9 会连 1e-12 幅度的非法扰动一起收下。
        #: 安全窗口是**测出来的**（`summary.safe_window`），不在配置里硬编码。
        "tolerance": 1e-9,
    },

    # ---- 搜索 ----
    "search": {
        "max_depth": 3,
        "budget_per_searcher": 120,
        #: 紧预算：用来**区分策略**的预算。
        #: 主预算 120 已经把空间覆盖到「随机搜索也饱和」的程度（实测 40~60 次即饱和），
        #: 在饱和点比策略等于比运气。10~30 次才是策略真正起作用的地方。
        "tight_budget": 20,
        "beam_width": 4,
        "population": 12,
        "mutation_rate": 0.35,
        "searchers": ["random", "greedy", "beam", "evolution"],
        "reference_pad": 0.05,
        # 计划里规则的**规范顺序**（只用于 oracle 的子集枚举与报告排序）
        "rule_order": list(LEGAL_NAMES),
        # 保险丝：一次 `run` 里最多施加多少次重写（防某个计划不收敛）
        "max_moves": 24,
    },

    # ---- 约束处理策略 ----
    "constraints": {
        "strategy": "reject",          # reject | penalty
        "penalty_weight": 1.0,
        #: 违反量的**固定**分母（`violation = max(0, drift−τ)/drift_scale`）。
        #: **不是「除以容差」** —— 除以 τ 会让惩罚在 τ→0 时退化成硬约束，
        #: 于是「软约束不保证可行」这条结论永远暴露不出来。
        #: 取 1.0 = 「漂移本身」就是量纲：逐位等价的重写违反量 ~1e-16（几乎不罚），
        #: 非法重写违反量 ~1.0（重罚）。**这个取值就是惩罚软硬的旋钮。**
        "drift_scale": 1.0,
    },

    # ---- 阶梯：逐级放松**一条**约束（容差） ----
    "ladder": {
        #: **分辨率放在信号所在的地方。** 合法重写的漂移是 7.5e-17 / 1.5e-16，
        #: 所以底部三级（5e-17 / 1e-16 / 2e-16）才看得见台阶；
        #: 上面的六级是刻意留的**平坦段** —— 平坦本身是读数
        #: （「τ 到 2e-16 就已经全部可行」= 阶梯到那里就用完了）。
        "tolerances": [0.0, 5e-17, 1e-16, 2e-16, 1e-15, 1e-12, 1e-9, 1e-6, 1e-3, 1e-1],
    },

    # ---- 预算扫描：单一预算点不是比较，是轶事 ----
    "budget_sweep": {
        "budgets": [10, 20, 30, 40, 60, 80, 120],
        "strategy": "reject",
        "tolerance": 0.0,
    },

    # ---- 对抗：注入的非法重写幅度 ----
    "adversarial": {
        "bias_epsilons": [1e-12, 1e-9, 1e-6, 1e-3, 1e-1],
    },

    # ---- 成本模型常数：**全部是假设**，不是测量（附录 C 276） ----
    "cost": {
        "launch_overhead_us": 3.0,
        "flops_per_us": 2000.0,
        "bytes_per_us": 400.0,
        "bytes_per_elem": 4,
    },
    "cost_sweep": {
        "launch_overhead_us": [0.0, 0.3, 1.0, 3.0, 10.0, 30.0],
    },

    # ---- 验收门槛：**每一条都必须能红** ----
    #
    # 写门槛时问两个问题：① 它在什么情况下会红？② 那个情况是真的错，还是运气不好？
    # 答不上来的那一条删掉 —— 一条恒真的判据与没有判据在输出里长得一模一样。
    # （第一版写过 `max_illegal_acceptance_below_cliff`：非法变体在「低于自己漂移的
    # 容差」下被接受 —— 按定义不可能发生。**它永远是绿的。** 已删。）
    "acceptance": {
        # -- 结构性：空间本身得是完整的
        "min_ladder_rungs": 10,
        "max_structural_failures": 0,
        "min_lineage_nodes": 20,
        "min_distinct_plans": 40,
        "min_order_sensitive_pairs": 1,
        # -- 重写的等价性分类：必须**两类都存在**
        "min_bit_identical_rewrites": 5,
        "min_inexact_rewrites": 3,
        # -- 容差崖：安全窗口非空（max 合法漂移 < min 非法漂移）
        "safe_window_nonempty": True,
        # -- 阶梯：**质量不动，可行域必须动**
        #: 严格扩张的级数（`n_feasible_space` 增大）。
        #: 质量那一列实测是**全平**的（见 `ladder_improves_quality` 那条期望被推翻），
        #: 所以阶梯「动没动」只能量在可行域上 —— 这是它能红的唯一读法。
        "min_ladder_feasible_steps": 2,
        # -- 前沿与搜索器
        #: 门槛落在**不同的目标向量个数**上，不落在计划数上 ——
        #: 实测：12 个前沿计划只有 2 个不同的权衡点，其余是「换个顺序得到同一张图」。
        #: 按计划数量，这条门槛会靠重复点通过。
        "min_frontier_points": 2,
        "max_greedy_hv_gap_vs_oracle": 0.15,
        #: 在**紧预算**下，至少有一个非随机搜索器明显优于随机。
        #: 主预算下这条**不成立**（随机已饱和）—— 那是 `searchers_beat_random`
        #: 那条期望被推翻的原因，而它正是「策略的贡献与预算的贡献分不开」的证据。
        "min_searcher_hv_gain_over_random_at_tight_budget": 0.05,
        "oracle_is_optimal": True,
        # -- 硬约束的交付判据：交出去的答案必须可行
        "max_answer_infeasible_rate": 0.0,
        # -- 阶梯单调（数学必然，红了就是实现错了）
        "min_monotone_strategies": 2,
        # -- 对抗用例（**不可调的不变量断言**）不许被偷偷删掉。
        #: ⚠️ 口径：「对抗用例」指的是 `invariants.py` 里那些不变量断言，
        #: **不是** `adversarial.json` 里那 7 个非法变体 ——
        #: 后者是被测量的**实验对象**（数据）。第一版把两者混称，
        #: 于是报告里写「7 条对抗用例」，而 L5 各项目是 24~38 条。
        #: 这条门槛量的是**注册表大小**（不跑用例就知道），所以它能在
        #: `cases` 被填进去之前就生效。
        "min_adversarial_cases": 44,
        # -- 保险丝：验收表本身不许被偷偷缩短
        "min_acceptance_checks": 19,
    },

    # ---- 设计期望（**允许被实测推翻**，被推翻的那条最值钱） ----
    "expectations": {
        "exact_equality_accepts_all_legal_rewrites": {
            "claim": "τ=0 时全部合法重写都能通过（即所有合法重写都逐位等价）",
            "expect": False,
        },
        "fusion_is_bit_identical": {
            "claim": "算子融合逐位等价（融合买到的是访存与 launch，不是算法）",
            "expect": True,
        },
        "order_matters": {
            "claim": "重写顺序影响结果（存在一对计划，规则相同、顺序不同、结果不同）",
            "expect": True,
        },
        "greedy_is_myopic": {
            "claim": "按单步收益贪心会比穷举最优差",
            "expect": True,
        },
        "ladder_is_monotone": {
            "claim": "容差放松 ⇒ 可行域扩张 ⇒ oracle 最优目标不减（数学必然）",
            "expect": True,
        },
        "safe_tolerance_window_is_nonempty": {
            "claim": "存在一个容差区间：全部合法重写通过、全部非法重写被拒",
            "expect": True,
        },
        "ladder_has_flat_top": {
            "claim": "阶梯在高容差端完全平坦（可行域一旦饱和，再放松容差没有收益）",
            "expect": True,
        },
        "ladder_improves_quality": {
            "claim": "放宽容差能改善答案质量（前沿变宽或最优点变高）",
            "expect": True,
        },
        "launch_overhead_dominates": {
            "claim": "融合的收益主要来自 launch 开销（把 launch_overhead_us 扫到 0 后收益大幅缩小）",
            "expect": True,
        },
        "objectives_are_independent": {
            "claim": "三个目标轴互不冗余（两两 |corr| < 0.95）",
            "expect": True,
        },
        "peak_memory_is_an_independent_axis": {
            "claim": "峰值内存是独立的一维（不被速度轴决定）",
            "expect": True,
        },
        "searchers_beat_random": {
            "claim": "在**主预算**下，非随机搜索器都优于随机搜索",
            "expect": True,
        },
        "searchers_beat_random_at_tight_budget": {
            "claim": "在**紧预算**下，至少有一个非随机搜索器明显优于随机搜索",
            "expect": True,
        },
        "random_saturates_at_main_budget": {
            "claim": "主预算下随机搜索已拿到 oracle 的全部超体积（⇒ 主预算不是比策略的地方）",
            "expect": True,
        },
        "soft_constraint_changes_the_answer": {
            "claim": "把硬约束换成软约束会改变交出去的答案",
            "expect": True,
        },
        "analytic_latency_tracks_wallclock": {
            "claim": "解析延迟与实测求值时间的排序同向（秩相关 ≥ 0.5）",
            "expect": True,
        },
        "validator_catches_all_illegal": {
            "claim": "校验器拒掉全部注入的非法重写（含 1e-12 幅度的那一档）",
            "expect": False,
        },
        "penalty_keeps_answers_feasible": {
            "claim": "软约束（penalty）交出去的答案仍然可行",
            "expect": False,
        },
    },
}


class ConfigError(Exception):
    """配置不合法。**一律抛错**，不静默回落（附录 C 2/3：宽松的 load_config
    会把「路径错了」伪装成「用默认值」）。"""


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并；**列表整体替换**（这是坑 204 的来源，所以有 `no_stale_config_keys`）。"""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path | None = None, *, root: str | Path | None = None) -> dict[str, Any]:
    """加载配置。

    `root` **真的参与路径解析** —— 不是「调用点写没写 `root=`」而是
    「这个参数有没有被用上」（附录 C 判据）。`--config` 的默认值是非 None 的
    `"configs/default.json"`，所以只在 `path is None` 那一支用 `root` 是**死代码**。
    """
    if path is None:
        base = Path(root) if root is not None else Path.cwd()
        p = base / "configs" / "default.json"
    else:
        p = Path(path)
        if not p.is_absolute() and root is not None:
            p = Path(root) / p
    if not p.exists():
        raise ConfigError(f"配置文件不存在：{p}")
    try:
        disk = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ConfigError(f"配置文件不是合法 JSON：{p} —— {e}") from e
    cfg = deep_merge(DEFAULTS, disk)
    cfg["_config_path"] = str(p)
    cfg["_root"] = str(Path(root) if root is not None else Path.cwd())

    # ⚠️ 这条保险丝**必须真的被调用**。第一版写好了 `no_stale_config_keys`
    #    却没接在这里，后果是：从 `DEFAULTS` 里删掉的门槛键（比如那条恒真的
    #    `max_illegal_acceptance_below_cliff`）**继续活在磁盘配置里**，
    #    被 `deep_merge` 带进运行配置，于是验收表里多出一条「没有任何检查函数
    #    认领」的门槛 —— 它永远是红的，而真因（配置陈旧）完全看不出来。
    #    **一个写了但没人调的检查，与没有这个检查在输出里完全一样。**
    stale = no_stale_config_keys(disk, DEFAULTS)
    if stale:
        raise ConfigError(
            "配置里有代码中不存在的键（多半是从 DEFAULTS 删了键但磁盘配置没跟着更新）：\n"
            + "\n".join(f"  - {k}" for k in stale)
            + f"\n  配置文件：{p}\n"
            "  修法：用 `graphoptfactory.config.dump_config(DEFAULTS, path)` 重新导出。"
        )
    validate_config(cfg)
    return cfg


def no_stale_config_keys(disk: dict[str, Any], defaults: dict[str, Any],
                         prefix: str = "") -> list[str]:
    """磁盘上出现了代码里没有的键 ⇒ 报出来（保险丝 `no_stale_config_keys`）。

    成因：`deep_merge` 对字典是递归合并，于是**从 `DEFAULTS` 删掉的键
    会从已落盘的配置里活下来**，代码里的删除根本传不到磁盘（附录 C 220）。
    """
    stale: list[str] = []
    for k, v in disk.items():
        here = f"{prefix}{k}"
        if k not in defaults:
            stale.append(here)
        elif isinstance(v, dict) and isinstance(defaults[k], dict):
            stale.extend(no_stale_config_keys(v, defaults[k], here + "."))
    return stale


def validate_config(cfg: dict[str, Any]) -> None:
    """配置校验。**每条报错写清「为什么这是错的」** —— 文案本身是被断言的一部分。"""
    errs: list[str] = []

    g = cfg["graph"]
    if int(g["n_layers"]) < 1:
        errs.append("graph.n_layers 必须 ≥ 1（否则主干上没有任何融合靶子）")
    if int(g["rows"]) < 1 or int(g["dim"]) < 1:
        errs.append("graph.rows / graph.dim 必须 ≥ 1")

    tol = float(cfg["semantics"]["tolerance"])
    if tol < 0:
        errs.append("semantics.tolerance 不能为负（负容差会让任何重写都不可行）")

    s = cfg["search"]
    if int(s["max_depth"]) < 1:
        errs.append("search.max_depth 必须 ≥ 1")
    if int(s["max_depth"]) > len(LEGAL_NAMES):
        errs.append(f"search.max_depth ≤ 合法规则条数 {len(LEGAL_NAMES)}"
                    "（计划里的规则不许重复，深度超过规则数就一定有重复）")
    if int(s["budget_per_searcher"]) < 1:
        errs.append("search.budget_per_searcher 必须 ≥ 1")
    if int(s["beam_width"]) < 1:
        errs.append("search.beam_width 必须 ≥ 1")
    if int(s["population"]) < 2:
        errs.append("search.population 必须 ≥ 2（进化搜索至少要两个个体才能交叉）")
    unknown = [x for x in s["rule_order"] if x not in LEGAL_NAMES]
    if unknown:
        errs.append(f"search.rule_order 含未知规则 {unknown}")
    if sorted(s["rule_order"]) != sorted(LEGAL_NAMES):
        errs.append("search.rule_order 必须恰好是全部合法规则的一个排列"
                    "（少一条 = 那条规则永远不会被 oracle 枚举到）")
    if int(s["max_moves"]) < int(s["max_depth"]):
        errs.append("search.max_moves 必须 ≥ max_depth（否则保险丝比计划本身还短）")

    if cfg["constraints"]["strategy"] not in ("reject", "penalty"):
        errs.append("constraints.strategy 只能是 reject / penalty")
    if float(cfg["constraints"]["penalty_weight"]) < 0:
        errs.append("constraints.penalty_weight 不能为负")

    tolerances = [float(x) for x in cfg["ladder"]["tolerances"]]
    if len(tolerances) < 3:
        errs.append("ladder.tolerances 至少 3 级（少了看不出阶梯）")
    if tolerances != sorted(tolerances):
        errs.append("ladder.tolerances 必须单调不减（阶梯的方向就是放松约束）")
    if 0.0 not in tolerances:
        errs.append("ladder.tolerances 必须含 0.0 —— 那是「只接受逐位等价」那一级")

    eps = [float(x) for x in cfg["adversarial"]["bias_epsilons"]]
    if len(eps) < 2:
        errs.append("adversarial.bias_epsilons 至少 2 档（否则定位不出容差崖）")
    if any(e <= 0 for e in eps):
        errs.append("adversarial.bias_epsilons 必须全部 > 0"
                    "（ε=0 是「没有扰动」，那不是一条非法重写）")
    if eps != sorted(eps):
        errs.append("adversarial.bias_epsilons 必须严格递增"
                    "（乱序的扫描表会印出一张假的单调表）")
    if max(eps) < max(tolerances):
        errs.append("adversarial.bias_epsilons 的最大值必须 ≥ 最大容差，"
                    "否则连最大那一级容差都没有被最大的扰动考验过")
    # ⚠️ **这里刻意不断言 `max(eps) > max(tolerances)`。**
    # 是否被接受取决于**实测漂移**（本项目的 `bias_epsilon` 实测 drift ≈ 1.519·ε），
    # 而配置校验只看得到 ε 与 τ 两个数、量不出漂移。在只能看见代理量的地方
    # 断言代理量之间的关系，等于**在配置里硬编码一个没人测量过的系数** ——
    # 改一次图结构，这条校验就变成一句假话，而且它永远是绿的。
    # 真正的断言（「ε 网格必须在崖的两侧都有格子」）写在 `adversarial.py`，
    # 那里漂移已经被算出来了。**校验只该断言它量得到的东西。**

    c = cfg["cost"]
    for k in ("launch_overhead_us", "flops_per_us", "bytes_per_us", "bytes_per_elem"):
        if float(c[k]) <= 0:
            errs.append(f"cost.{k} 必须 > 0（0 会让对应那一维从成本里消失）")

    acc = cfg["acceptance"]
    if int(acc["min_frontier_points"]) < 2:
        errs.append("acceptance.min_frontier_points 必须 ≥ 2"
                    "（1 个点不是前沿，是一个点）")
    if float(acc["max_answer_infeasible_rate"]) != 0.0:
        errs.append("acceptance.max_answer_infeasible_rate 必须是 0.0"
                    "—— 交出去的答案必须可行，这是硬约束的定义")
    if not bool(acc["safe_window_nonempty"]):
        errs.append("acceptance.safe_window_nonempty 必须为 True"
                    "—— 关掉它等于声明「不存在安全容差窗口」，"
                    "而那会让「容差该取多少」这个问题失去答案")
    if not bool(acc["oracle_is_optimal"]):
        errs.append("acceptance.oracle_is_optimal 必须为 True"
                    "—— oracle 穷举了整个空间，它不是最优就说明枚举或评估有 bug")
    if int(acc["min_acceptance_checks"]) < 10:
        errs.append("acceptance.min_acceptance_checks 必须 ≥ 10"
                    "（这条保险丝自己也不许被调松）")

    exp = cfg["expectations"]
    for name, spec in exp.items():
        if not isinstance(spec, dict) or "claim" not in spec or "expect" not in spec:
            errs.append(f"expectations.{name} 必须有 claim 与 expect 两个字段")

    if errs:
        raise ConfigError("配置不合法：\n  - " + "\n  - ".join(errs))


def protocol_digest(cfg: dict[str, Any]) -> str:
    """协议指纹：**只覆盖影响结论的字段**，不含路径。

    两处 `run` 用同一份配置 ⇒ 同一个协议指纹。加了 `_config_path` 进去会让
    「同一份协议、两个不同的调用路径」得到两个指纹，而那正是我们要它相同的情形。
    """
    payload = {k: v for k, v in cfg.items() if not k.startswith("_")}
    return digest_of(json.dumps(payload, sort_keys=True, ensure_ascii=False))


def dump_config(cfg: dict[str, Any], path: str | Path) -> None:
    """把当前生效的配置落盘（`DEFAULTS` 的副本 + 覆盖）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in cfg.items() if not k.startswith("_")}
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                 encoding="utf-8", newline="\n")
