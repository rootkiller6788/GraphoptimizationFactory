"""报告渲染：**只吃产物字典**，不读运行期对象。

## 这条纪律买到两样

1. `visualize` 能从 `out/` 重建报告，一行都不重跑流水线；
2. 它证明的东西更强：从「流水线是确定的」变成「**产物足以重造报告**」。

配一条对抗用例：把一次 run 折成产物 → 从产物重建 → 再渲染，
与直接渲染**逐位比**。这条用例在 `L5-quantfactory` 抓出过
`render_html` 里 `len(run.products) - 2` 在产物还没填满时印出「其余 −2 个文件」——
**负数比报错更危险，因为它看起来像个数字。**

## Markdown 与 HTML 共用一份文档结构

`build_doc(products)` 产出 `Block` 列表，两种格式各自渲染它。
分两套实现的话，两份报告会慢慢长歪，而没有任何地方会红。
"""

from __future__ import annotations

import html as _html
from typing import Any, Sequence

Block = tuple

#: 目标轴的展示名与「越大越好」的方向提示。
_AXIS = {
    "speedup_gain": "延迟下降",
    "memory_saving": "内存下降",
    "launch_reduction": "launch 下降",
}


def _f(x: Any, nd: int = 6) -> str:
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "是" if x else "否"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _sci(x: Any) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        if x == float("inf"):
            return "∞"
        if x != x:
            return "nan"
        return f"{x:.3e}"
    return str(x)


def _pct(x: Any, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and x != x):
        return "—"
    return f"{100.0 * float(x):.{nd}f}%"


# --------------------------------------------------------------------- 文档结构
def build_doc(products: dict[str, Any]) -> list[Block]:
    cfg = products["config.json"]["resolved"]
    base = products["baseline.json"]
    space = products["space.json"]
    lin = products["lineage.json"]
    adv = products["adversarial.json"]
    ladder = products["ladder.json"]
    search = products["search.json"]
    bench = products["benchmark.json"]
    exp = products["expectations.json"]
    acc = products["acceptance.json"]
    summ = products["summary.json"]

    b: list[Block] = []

    # ---------------------------------------------------------------- 1 概览
    b.append(("h", 1, "GraphOptimizationFactory — 运行报告"))
    b.append(("p",
              "形状判定：**搜索闭环 · 第二形态（硬约束）**。"
              "候选是**重写序列**（有序、不重复），硬约束是**语义等价**，"
              "阶梯是**容差 τ 逐级放松**。"
              "每个候选都被**真跑一遍**：纯 Python 求值器把图算出来，"
              "语义等价是可执行的，不靠声明。"))
    b.append(("table", ["项", "值"], [
        ["协议指纹", products["config.json"]["protocol_digest"]],
        ["搜索空间", f"{search['space_size']} 个计划"
                     f"（8 规则、深度 ≤ {cfg['search']['max_depth']}）"],
        ["不同图指纹", str(space["n_distinct_digests"])],
        ["不同目标向量", str(space["n_distinct_objective_vectors"])],
        ["搜索器", "、".join(search["searchers"])],
        ["预算", f"oracle {search['budget_asymmetry']['oracle']} / "
                 f"其余 {search['budget_asymmetry']['others']}"],
        ["约束策略", "、".join(["reject", "penalty"])],
        ["阶梯容差", "、".join(f"{t:g}" for t in cfg["ladder"]["tolerances"])],
        ["默认容差", f"{cfg['semantics']['tolerance']:g}"],
        ["缓存（唯一 / 命中）", f"{search['cache']['unique_evaluations']} / "
                               f"{search['cache']['cache_hits']}"],
    ]))

    # ---------------------------------------------------------------- 2 基线
    m = base["metrics"]
    b.append(("h", 2, "基线图"))
    b.append(("table", ["读数", "值"], [
        ["节点数", str(m["nodes"])],
        ["算子分布", "、".join(f"{k}×{v}" for k, v in base["op_counts"].items())],
        ["launch 次数", str(m["launches"])],
        ["FLOPs", str(m["flops"])],
        ["搬运字节", str(m["bytes_moved"])],
        ["解析延迟 (µs)", _f(m["latency_us"], 3)],
        ["峰值内存 (字节)", str(m["peak_bytes"])],
        ["常量节点", str(m["const_nodes"])],
        ["图指纹", base["digest"]],
    ]))
    b.append(("p", "成本读数的来源标签（**哪些是结构性的、哪些是解析算出来的**）："
              + "；".join(f"`{k}`={v}" for k, v in base["cost_source"].items())))

    # ---------------------------------------------------------------- 3 血缘
    b.append(("h", 2, "重写血缘（深度 1）"))
    rows = []
    for r in lin["rules"]:
        d = r["delta"]
        rows.append([
            r["rule"],
            "合法" if r["kind"] == "legal" else "**非法**",
            str(r["hits"]),
            _sci(r["drift"]),
            "逐位等价" if r["bit_identical"] else "近似",
            f"{d['nodes']:+d}",
            f"{d['launches']:+d}",
            _f(d["latency_us"], 3),
            f"{d['peak_bytes']:+d}",
        ])
    b.append(("table",
              ["规则", "类别", "命中", "漂移", "等价性", "Δ节点", "Δlaunch",
               "Δ延迟", "Δ峰值内存"], rows))
    b.append(("p", f"血缘图：{lin['graph']['n_nodes']} 个不同图指纹、"
                   f"{lin['graph']['n_edges']} 条边、{lin['graph']['n_plans']} 个计划。"
                   "**节点是图指纹，不是计划** —— 于是「不同计划得到同一张图」"
                   "这件事看得见。"))

    o = lin["order_sensitivity"]
    b.append(("h", 3, "顺序敏感性"))
    b.append(("p", f"{o['n_order_sensitive_unordered']}/{o['n_unordered_pairs']} "
                   "个无序规则对换序后结果不同。"
                   "**顺序是空间的一部分** —— 否则「先融合再合并」写进规则里就够了，"
                   "不需要搜。"))
    if o["examples"]:
        b.append(("table", ["先 A 后 B", "A→B 目标", "B→A 目标", "最大分量差"], [
            [" / ".join(e["a_then_b"]),
             "(" + ", ".join(_f(v, 4) for v in e["values_ab"]) + ")",
             "(" + ", ".join(_f(v, 4) for v in e["values_ba"]) + ")",
             _f(e["max_abs_diff"], 6)]
            for e in o["examples"][:6]
        ]))

    # ---------------------------------------------------------------- 4 对抗
    b.append(("h", 2, "对抗：容差崖与安全窗口"))
    w = exp["derived"]["safe_window"]
    b.append(("p",
              f"安全窗口 = **[{_sci(w['lo'])}, {_sci(w['hi'])})**，"
              f"宽 {_f(w['decades'], 2)} 个数量级"
              f"（下界来自 `{w['lo_from']}`）。"
              f"默认容差 τ={w['default_tolerance']:g} "
              f"**{'落在窗口内' if w['default_inside'] else '落在窗口外'}**。"))
    b.append(("table",
              ["非法变体", "ε", "漂移", "漂移/ε", "首次被收下的容差"]
              + [f"τ={t:g}" for t in cfg["ladder"]["tolerances"]],
              [[v["variant"], "—" if v["epsilon"] is None else f"{v['epsilon']:g}",
                _sci(v["drift"]),
                "—" if v.get("drift_per_epsilon") is None
                else _f(v["drift_per_epsilon"], 6),
                "从未" if v["cliff_tolerance"] is None
                else f"{v['cliff_tolerance']:g}"]
               + ["收下" if v["accepted_at"][str(t)] else "拒" 
                  for t in cfg["ladder"]["tolerances"]]
               for v in adv["variants"]]))
    b.append(("p",
              f"**默认容差下漏掉 {adv['n_accepted_at_default_tolerance']} 个非法变体**："
              f"{adv['accepted_at_default_tolerance'] or '（无）'}。"
              "这不是 bug，是**配置取值的后果** —— "
              "「校验器能不能挡住非法重写」这个问题的答案取决于容差取多少，"
              "而容差是配置里的一个数。**不看容差的「校验器通过率」是没有意义的读数。**"))
    if adv.get("objective_deltas"):
        b.append(("table", ["非法变体（目标读数）", "延迟下降", "内存下降", "launch 下降"], [
            [r["variant"]] + [_pct(x) for x in r["objectives"]]
            for r in adv["objective_deltas"]
        ]))
        b.append(("p", "**非法重写看起来有多好**，必须一起报 —— "
                       "否则「不放松容差」会被读成一个纯收益，而它是有代价的。"))

    # ---------------------------------------------------------------- 5 阶梯
    b.append(("h", 2, "阶梯：容差逐级放松"))
    for strat in ("reject", "penalty"):
        rows = [r for r in ladder["rungs"] if r["strategy"] == strat]
        rows.sort(key=lambda r: r["tolerance"])
        b.append(("h", 3, f"策略 = {strat}"))
        b.append(("table",
                  ["τ", "可行计划数", "oracle 最好均值", "oracle 前沿点数",
                   "oracle 超体积", "参考点"],
                  [[f"{r['tolerance']:g}", str(r["n_feasible_space"]),
                    _f(r["best_mean_gain"].get("oracle"), 6),
                    str(r["cells"]["oracle"]["n_frontier"]),
                    _f(r["cells"]["oracle"]["hypervolume"], 8),
                    "(" + ", ".join(_f(x, 4) for x in r["reference"]) + ")"]
                   for r in rows]))
    for mo in ladder["monotonicity"]:
        b.append(("p",
                  f"单调性（{mo['strategy']}）："
                  + ("**单调不减** —— τ 放松 ⇒ 可行域扩张 ⇒ 上确界不减，这是数学必然，"
                     "红了就是实现错了。" if mo["monotone"]
                     else f"**出现 {len(mo['violations'])} 处下降**：{mo['violations']}")))

    # ---------------------------------------------------------------- 6 搜索器
    b.append(("h", 2, "搜索器对比（τ=0 / reject）"))
    tau0 = min(float(x) for x in cfg["ladder"]["tolerances"])
    rung0 = next(r for r in ladder["rungs"]
                 if r["strategy"] == "reject" and r["tolerance"] == tau0)
    cells: dict[str, Any] = dict(rung0["cells"])
    b.append(("table",
              ["搜索器", "预算", "实际用", "重复提案", "搭车", "可行", "前沿计划",
               "不同权衡点", "超体积", "相对 oracle 缺口", "最好均值"],
              [[n, str(c["budget"]), str(c["used"]), str(c["hits"]), str(c["shared"]),
                str(c["n_feasible"]), str(c["n_frontier"]),
                str(c["n_frontier_distinct"]),
                _f(c["hypervolume"], 8),
                ("—" if c["hv_gap_vs_oracle"] is None
                 else _pct(c["hv_gap_vs_oracle"])),
                _f((c["best"] or {}).get("objectives") and
                   sum(c["best"]["objectives"]) / 3, 6)]
               for n, c in sorted(cells.items())]))
    b.append(("p",
              "**「前沿计划」与「不同权衡点」是两个数，都要报。**"
              f"实测：oracle 的 {cells['oracle']['n_frontier']} 个前沿计划只有 "
              f"**{cells['oracle']['n_frontier_distinct']} 个不同的目标向量** —— "
              "其余是「换个顺序得到同一张图」。按计划数报前沿大小会被顺序冗余灌水，"
              "而门槛也会靠重复点通过。"))
    b.append(("p",
              "**预算是不对称的，而且必须显式报出来**：oracle 拿整个空间（400），"
              "其余拿 120。这不是特权，是定义 —— 若把 oracle 也压在 120，"
              "它交出的就是「120 个里的最优」，而它就不再是 oracle。"
              "「搭车」= 提案已被别的搜索器评估过（省了算力，不省预算）。"))
    b.append(("p", "搜索器声明的超参：" + "；".join(
        f"`{k}`={list(v)}" for k, v in search["params_declared"].items())))

    # ------------------------------------------------------------- 6b 预算扫描
    sw = products["budget_sweep.json"]
    b.append(("h", 3, "预算扫描：**单一预算点不是比较，是轶事**"))
    b.append(("p",
              f"主预算 {cfg['search']['budget_per_searcher']}、"
              f"紧预算 {sw['tight_budget']}。"
              f"**随机搜索在 {sw['random_saturation_budget']} 次评估就把 oracle 的"
              "超体积拿满了** —— 主预算下的「谁更强」在比运气，"
              "而 `beam` 因为「展开到一半预算就用完」显得最差，"
              "那个排名是**预算截断位置**的产物，不是策略能力的读数。"))
    b.append(("table",
              ["预算"] + list(sw["searchers"]) + ["oracle"],
              [[str(r["budget"])]
               + [_f(r["cells"][n]["hypervolume"], 6) for n in sw["searchers"]]
               + [_f(r["oracle_hypervolume"], 6)]
               for r in sw["rows"]]))
    b.append(("table",
              ["预算", "实际用量（各搜索器）", "最好的非随机", "相对随机增益"],
              [[str(r["budget"]),
                " / ".join(f"{n}:{r['cells'][n]['used']}" for n in sw["searchers"]),
                m["best_searcher"], _pct(m["gain_over_random"])]
               for r, m in zip(sw["rows"], sw["margins"])]))
    b.append(("p", "**策略的贡献与预算的贡献是耦合的** —— 最优点随预算换人。"
                   "这个结论只能从曲线上读出来；只报一个预算点，"
                   "等于替读者选了「在哪个预算下比较」这个前提。"))

    # ---------------------------------------------------------------- 7 前沿
    key = f"reject@{tau0:g}"
    fr = ladder["frontier"].get(key, {})
    b.append(("h", 2, f"交付物：τ={tau0:g} / reject 的 Pareto 前沿"))
    b.append(("p", "**交付物是前沿，不是点。**结果层绝不允许标量化 —— "
                   "一旦把三轴压成一个数，输出就不再是前沿。"
                   "下面的 `均值` 只是一个**声明的投影**，用来排序与检查单调性。"))
    b.append(("table",
              ["计划", "漂移", "延迟下降", "内存下降", "launch 下降", "均值",
               "节点", "launch", "延迟 µs", "峰值字节"],
              [[" → ".join(p["rules"]), _sci(p["drift"])]
               + [_pct(x) for x in p["objectives"]]
               + [_f(p["mean_gain"], 6), str(p["nodes"]), str(p["launches"]),
                  _f(p["latency_us"], 3), str(p["peak_bytes"])]
               for p in fr.get("points", [])]))
    b.append(("svg", _frontier_svg(fr), "前沿的三对二维投影（参考点之外的点不画）"))
    best = ladder.get("best_answer") or {}
    if best:
        b.append(("p",
                  f"**头条数字**（τ=0 / reject，oracle 按均值挑的代表点）："
                  f"`{' → '.join(best['rules'])}`，"
                  f"延迟下降 **{_pct(best['objectives'][0])}**、"
                  f"内存下降 **{_pct(best['objectives'][1])}**、"
                  f"launch 下降 **{_pct(best['objectives'][2])}**，"
                  f"漂移 {_sci(best['drift'])}（**逐位等价**）。"))

    # ---------------------------------------------------------------- 8 期望
    b.append(("h", 2, "设计期望 vs 实测"))
    b.append(("p", f"共 {exp['n']} 条：**{exp['n_held']} 条成立、"
                   f"{exp['n_refuted']} 条被推翻、{exp['n_unmeasurable']} 条算不出来**。"
                   "被推翻的那几条最值钱 —— 它们是「原来想错了什么」。"
                   f"另有 {exp['n_timing']} 条**依赖实测耗时**的期望，"
                   "它们的读数随机器变，所以不在这个文件里"
                   "（见 `benchmark.json` 的 `timing_expectations`）。"))
    b.append(("table", ["期望", "原期望", "实测", "裁定", "读数"], [
        [i["name"], _f(i["expect"]), _f(i["observed"]),
         {"held": "成立", "refuted": "**被推翻**", "unmeasurable": "算不出来"}[i["verdict"]],
         i["detail"]]
        for i in exp["items"]]))
    timing = bench.get("timing_expectations") or []
    if timing:
        b.append(("h", 3, "时序类期望（**不进版本摘要**）"))
        b.append(("p", "它们的读数来自实测耗时 —— 同一份输入跑两次会得到不同的数。"
                       "把它们放进版本摘要，版本号就失去意义。"))
        b.append(("table", ["期望", "原期望", "实测", "裁定", "读数"], [
            [i["name"], _f(i["expect"]), _f(i["observed"]),
             {"held": "成立", "refuted": "**被推翻**",
              "unmeasurable": "算不出来"}[i["verdict"]], i["detail"]]
            for i in timing]))

    # ---------------------------------------------------------------- 9 验收
    b.append(("h", 2, "验收"))
    b.append(("p", f"{acc['n']} 条门槛：**{acc['n_passed']} 通过、{acc['n_failed']} 未通过**。"
              + ("" if acc["n_failed"] == 0
                 else f"未通过：{acc['failed']}")))
    b.append(("table", ["门槛", "阈值", "实测", "结论", "说明"], [
        [c["name"], str(c["threshold"]), _f(c["observed"]),
         "通过" if c["passed"] else "**未通过**", c["detail"]]
        for c in acc["checks"]]))

    # 9b 对抗用例：**不可调**的不变量断言（与门槛分工不同）
    if acc.get("cases"):
        n_cases = int(acc["n_cases"])
        n_bad = int(acc["n_cases_failed"])
        b.append(("h", 3, "对抗用例（不可调的不变量断言）"))
        b.append(("p", f"{n_cases} 条：**{n_cases - n_bad} 成立、{n_bad} 不成立**。"
                  "门槛是**达标线**（阈值写在配置里，可以被调松）；"
                  "这一节是**不变量**（配置里没有它的阈值）——"
                  "红了说明代码或产物自相矛盾，不是「这次没达标」。"
                  + ("" if n_bad == 0
                     else f" 不成立的：{acc['failed_cases']}")))
        b.append(("table", ["对抗用例", "结论", "说明"], [
            [c["name"], "成立" if c["ok"] else "**不成立**", c["detail"]]
            for c in acc["cases"]]))
        b.append(("p", "⚠️ 口径：`adversarial.json` 里那 "
                       f"{adv['n_illegal_variants']} 个**非法变体**是"
                       "**被测量的实验对象**（数据），**不是**对抗用例。"
                       "两者混称会让「这个项目有多少条对抗用例」这句话没有意义 ——"
                       "本项目第一版就是这么错的。"))

    # ---------------------------------------------------------------- 10 成本
    b.append(("h", 2, "成本模型：假设与敏感度"))
    b.append(("p", "成本模型**全部是算出来的**（`COST_SOURCE` 逐键标注）。"
                   "下面扫 `launch_overhead_us`，一次只动一个变量 —— "
                   "凡「某机制买到 X」的结论，都要报告它在哪个系数区间才成立。"))
    if bench.get("cost_sweep"):
        b.append(("table", ["launch_overhead_us", "基线延迟", "最好延迟", "speedup"], [
            [_f(r["launch_overhead_us"], 3), _f(r["baseline_latency_us"], 3),
             _f(r["best_latency_us"], 3), _f(r["speedup"], 6)]
            for r in bench["cost_sweep"]]))
    wc = bench.get("wallclock", {})
    b.append(("p",
              f"解析延迟 vs **实测耗时**的 Spearman ρ = {_f(wc.get('spearman_rho'), 6)}"
              f"（{wc.get('n_plans')} 个前沿计划，重复取最小）。"
              "这一条问的是「成本模型能不能被当作代理目标」—— "
              "ρ 低就说明搜索优化的是一张**假设的表**，不是真实耗时。"))
    if wc.get("rows"):
        b.append(("table", ["计划", "解析延迟 µs", "实测耗时 µs"], [
            [" → ".join(r["rules"]), _f(r["analytic_latency_us"], 3),
             _f(r["measured_us"], 3)] for r in wc["rows"]]))

    # ---------------------------------------------------------------- 11 产物
    b.append(("h", 2, "产物"))
    b.append(("p", "进版本摘要的产物（顺序即摘要顺序）："
              + "、".join(f"`{n}`" for n in products["version.json"]["versioned_files"])))
    b.append(("p", "落盘但不进摘要："
              + "、".join(f"`{n}`" for n in products["version.json"]["unversioned_files"])))

    # ---------------------------------------------------------------- 12 汇总
    b.append(("h", 2, "汇总"))
    b.append(("table", ["项", "值"], [
        ["版本", products["version.json"]["version"]],
        ["空间规模", str(summ["space_size"])],
        ["τ=0 前沿点数", str(summ["frontier_points_at_tau0"])],
        ["τ=0 超体积", _f(summ["hypervolume_at_tau0"], 8)],
        ["合法逐位等价重写", f"{summ['n_legal_bit_identical']} / "
                             f"{summ['n_legal_bit_identical'] + summ['n_legal_inexact']}"],
        ["顺序敏感无序对", str(summ["order_sensitive_unordered_pairs"])],
        ["安全窗口", f"[{_sci(summ['safe_window']['lo'])}, "
                     f"{_sci(summ['safe_window']['hi'])})"],
        ["默认容差漏掉的非法变体", str(summ["illegal_accepted_at_default_tolerance"])],
        ["期望 成立/推翻/算不出", f"{summ['expectations']['n_held']} / "
                                 f"{summ['expectations']['n_refuted']} / "
                                 f"{exp['n_unmeasurable']}"],
        ["验收 通过/总数", f"{summ['acceptance']['n_passed']} / "
                           f"{summ['acceptance']['n']}"],
    ]))
    return b


# --------------------------------------------------------------------- 前沿 SVG
def _frontier_svg(fr: dict[str, Any], size: int = 190) -> str:
    pts = fr.get("points", [])
    keys = ["speedup_gain", "memory_saving", "launch_reduction"]
    labels = [_AXIS[k] for k in keys]
    pairs = [(0, 1), (0, 2), (1, 2)]
    pad = 34
    w = pad * 3 + size * 3
    h = size + pad * 2 + 16
    parts = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" '
             f'font-family="system-ui,-apple-system,Segoe UI,sans-serif">']
    allv = [[p["objectives"][i] for p in pts] for i in range(3)]
    lo = [min(v) - 0.02 for v in allv] if pts else [0, 0, 0]
    hi = [max(v) + 0.02 for v in allv] if pts else [1, 1, 1]

    for pi, (ix, iy) in enumerate(pairs):
        ox = pad + pi * (size + pad)
        oy = pad + 8

        def sx(x, ix=ix, ox=ox):
            span = (hi[ix] - lo[ix]) or 1.0
            return ox + (x - lo[ix]) / span * size

        def sy(y, iy=iy, oy=oy):
            span = (hi[iy] - lo[iy]) or 1.0
            return oy + size - (y - lo[iy]) / span * size

        parts.append(f'<rect x="{ox}" y="{oy}" width="{size}" height="{size}" '
                     f'fill="#fbfbfd" stroke="#c9ccd6" stroke-width="1"/>')
        parts.append(f'<text x="{ox + size / 2:.0f}" y="{oy + size + 15}" '
                     f'text-anchor="middle" font-size="10" fill="#333">{labels[ix]}</text>')
        parts.append(f'<text x="{ox - 6}" y="{oy + size / 2:.0f}" text-anchor="middle" '
                     f'font-size="10" fill="#333" '
                     f'transform="rotate(-90 {ox - 6} {oy + size / 2:.0f})">'
                     f'{labels[iy]}</text>')
        for p in pts:
            parts.append(f'<circle cx="{sx(p["objectives"][ix]):.2f}" '
                         f'cy="{sy(p["objectives"][iy]):.2f}" r="3.2" '
                         f'fill="#2f6fd0" fill-opacity="0.85"/>')
        # 基线点 (0,0)
        parts.append(f'<path d="M {sx(0) - 4:.2f} {sy(0):.2f} L {sx(0) + 4:.2f} {sy(0):.2f} '
                     f'M {sx(0):.2f} {sy(0) - 4:.2f} L {sx(0):.2f} {sy(0) + 4:.2f}" '
                     f'stroke="#c0392b" stroke-width="1.4" fill="none"/>')
    parts.append('</svg>')
    return "".join(parts)


# --------------------------------------------------------------------- 渲染器
def to_markdown(doc: Sequence[Block]) -> str:
    out: list[str] = []
    for blk in doc:
        kind = blk[0]
        if kind == "h":
            out.append("#" * blk[1] + " " + blk[2])
        elif kind == "p":
            out.append(blk[1])
        elif kind == "ul":
            out.extend(f"- {x}" for x in blk[1])
        elif kind == "table":
            headers, rows = blk[1], blk[2]
            out.append("| " + " | ".join(str(x) for x in headers) + " |")
            out.append("|" + "|".join(" --- " for _ in headers) + "|")
            for r in rows:
                out.append("| " + " | ".join(str(x) for x in r) + " |")
        elif kind == "svg":
            out.append(f"> （图形见 `report.html`：{blk[2]}）")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def to_html(doc: Sequence[Block], title: str) -> str:
    body: list[str] = []
    for blk in doc:
        kind = blk[0]
        if kind == "h":
            body.append(f"<h{blk[1]}>{_inline(blk[2])}</h{blk[1]}>")
        elif kind == "p":
            body.append(f"<p>{_inline(blk[1])}</p>")
        elif kind == "ul":
            body.append("<ul>" + "".join(f"<li>{_inline(x)}</li>" for x in blk[1])
                        + "</ul>")
        elif kind == "table":
            headers, rows = blk[1], blk[2]
            th = "".join(f"<th>{_inline(x)}</th>" for x in headers)
            tr = "".join("<tr>" + "".join(f"<td>{_inline(x)}</td>" for x in r)
                         + "</tr>" for r in rows)
            body.append(f"<table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody>"
                        "</table>")
        elif kind == "svg":
            body.append(f"<figure>{blk[1]}<figcaption>{_inline(blk[2])}</figcaption>"
                        "</figure>")
    css = """
    :root { color-scheme: light; }
    body { margin: 0 auto; max-width: 1180px; padding: 32px 24px 64px;
           background: #ffffff; color: #1c1f26;
           font: 14px/1.7 system-ui, -apple-system, "Segoe UI", sans-serif; }
    h1 { font-size: 26px; margin: 0 0 6px; letter-spacing: -0.01em; }
    h2 { font-size: 19px; margin: 34px 0 10px; padding-bottom: 6px;
         border-bottom: 1px solid #e6e8ee; }
    h3 { font-size: 15px; margin: 22px 0 8px; color: #333a46; }
    p { margin: 8px 0; }
    code { background: #f2f4f8; border-radius: 4px; padding: 1px 5px;
           font: 12.5px/1.5 ui-monospace, SFMono-Regular, Consolas, monospace; }
    table { border-collapse: collapse; margin: 12px 0 18px; width: 100%;
            font-size: 13px; }
    th, td { border: 1px solid #e2e5ec; padding: 6px 9px; text-align: left;
             vertical-align: top; }
    th { background: #f6f8fb; font-weight: 600; }
    tbody tr:nth-child(even) { background: #fcfdff; }
    figure { margin: 14px 0 22px; }
    figcaption { font-size: 12.5px; color: #5a6070; margin-top: 6px; }
    svg { max-width: 100%; height: auto; }
    strong { font-weight: 650; }
    """
    return ("<!DOCTYPE html>\n<html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>{_html.escape(title)}</title><style>{css}</style></head><body>"
            + "\n".join(body) + "</body></html>\n")


def _inline(text: str) -> str:
    """极小的行内标记：`**粗体**`、`` `代码` ``。先转义再替换。"""
    s = _html.escape(str(text))
    while "**" in s:
        a = s.find("**")
        b = s.find("**", a + 2)
        if b < 0:
            break
        s = s[:a] + "<strong>" + s[a + 2:b] + "</strong>" + s[b + 2:]
    while "`" in s:
        a = s.find("`")
        b = s.find("`", a + 1)
        if b < 0:
            break
        s = s[:a] + "<code>" + s[a + 1:b] + "</code>" + s[b + 1:]
    return s


def render_markdown(products: dict[str, Any]) -> str:
    return to_markdown(build_doc(products))


def render_html(products: dict[str, Any]) -> str:
    v = products["version.json"]["version"]
    return to_html(build_doc(products), f"GraphOptimizationFactory 运行报告 {v}")
