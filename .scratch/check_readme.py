"""把 README 附录 B 里出现的每一个数字，从最终 out/ 重新导出并逐条比对。

判据：**README 里不许有一个数字是凭记忆写的。**
做法：每个 claim 都从产物现算，格式化成 README 里的写法，再检查它是 README 的子串。

比对前先把 README 归一化，否则会出一屏假阳性：
  · 去掉 `**` 加粗
  · 把 Unicode 减号 `−` 换成 ASCII `-`（README 里为排版用了 U+2212）
  · 把全角括号换成半角
每个 claim 可以给多个可接受写法（例如 `0.00%` 与 `0%`）。

用法：python .scratch/check_readme.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
README = (ROOT / "README.md").read_text(encoding="utf-8")


def norm(s: str) -> str:
    """README 与 needle 走同一套归一化 —— 只归一化一边必然出一屏假阳性。"""
    return (s.replace("**", "")
             .replace("\u2212", "-")
             .replace("（", "(").replace("）", ")"))


NORM = norm(README)


def load(name: str):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


P = {f.name: load(f.name) for f in OUT.glob("*.json")}

SP, LAD, LIN, ADV, BM = P["space.json"], P["ladder.json"], P["lineage.json"], P["adversarial.json"], P["benchmark.json"]
SW, SR, ACC, EXP, SUM, BL = P["budget_sweep.json"], P["search.json"], P["acceptance.json"], P["expectations.json"], P["summary.json"], P["baseline.json"]

R0 = next(r for r in LAD["rungs"] if r["strategy"] == "reject" and r["tolerance"] == 0.0)
MONO_REJECT = next(m for m in LAD["monotonicity"] if m["strategy"] == "reject")
VARIANTS = {v["variant"]: v for v in ADV["variants"]}
COST = {c["launch_overhead_us"]: c for c in BM["cost_sweep"]}
SW_ROW = {r["budget"]: r for r in SW["rows"]}
CHECK = {c["name"]: c for c in ACC["checks"]}
EXP_ITEM = {i["name"]: i for i in EXP["items"]}

claims: list[tuple[str, str, tuple[str, ...]]] = []


def claim(section: str, desc: str, *needles: str) -> None:
    """needle 与 README 走同一套归一化，否则 `**` 会造出一屏假阳性。"""
    claims.append((section, desc, tuple(norm(n) for n in needles)))


def f6(x: float) -> str:
    return f"{x:.6f}"


def pct(x: float) -> tuple[str, str]:
    """两种可接受写法：`6.25%`（不省尾零）与 `6.25%`/`0%`（省尾零）。"""
    return (f"{x * 100:.2f}%", f"{x * 100:g}%")


def e4(x: float) -> str:
    return f"{x:.4e}"


# ---------------------------------------------------------------- B.1
claim("B.1", "空间大小", f"**{SP['space_size']}** 个计划")
claim("B.1", "空间分解", "8+56+336")
claim("B.1", "不同图指纹", f"**{SP['n_distinct_digests']}**")
claim("B.1", "不同目标向量", f"**{SP['n_distinct_objective_vectors']}**")
claim("B.1", "血缘图规模", f"{LIN['graph']['n_nodes']} 节点 / **{LIN['graph']['n_edges']}** 边")
claim("B.1", "缓存", f"**{SR['cache']['unique_evaluations']}** 次（唯一）/ **{SR['cache']['cache_hits']}** 次缓存命中")
_t = BM["timings"]
_total = sum(_t[k] for k in ("lineage", "ladder", "budget_sweep", "adversarial", "benchmark"))
claim("B.1", "总耗时", f"**{_total:.2f} s**")
for k in ("lineage", "ladder", "budget_sweep", "adversarial", "benchmark"):
    claim("B.1", f"分段耗时 {k}", f"{_t[k]:.2f}", f"{_t[k]:.3f}")

# ---------------------------------------------------------------- B.2
FRONT0 = R0["cells"]["oracle"]
claim("B.2", "前沿计划数", f"**{FRONT0['n_frontier']} 个前沿计划")
claim("B.2", "不同权衡点", f"只有 {FRONT0['n_frontier_distinct']} 个不同的权衡点")
claim("B.2", "超体积", f"**{SP['hypervolume_at_tau0']:.12f}**")
_ref = "(" + ", ".join(f"{v:.12f}".rstrip("0").rstrip(".") for v in SP["reference"]) + ")"
claim("B.2", "参考点", _ref)
claim("B.2", "基线规模", f"{BL['metrics']['nodes']} 节点 / {BL['metrics']['launches']} launches / "
                        f"{BL['metrics']['latency_us']:.3f} µs / {BL['metrics']['peak_bytes']} 字节")
BA = LAD["best_answer"]
claim("B.2", "头条答案三目标", *[f"**{p}**" for o in BA["objectives"] for p in pct(o)])
claim("B.2", "头条答案延迟绝对值", f"{BL['metrics']['latency_us']:.3f}→{BA['latency_us']:.3f} µs")
claim("B.2", "头条答案内存绝对值", f"{BL['metrics']['peak_bytes']}→{BA['peak_bytes']} 字节")
claim("B.2", "头条答案节点", f"节点 {BL['metrics']['nodes']}→{BA['nodes']}")
claim("B.2", "头条答案 launch", f"launch {BL['metrics']['launches']}→{BA['launches']}")

# B.2 前沿表两行：从 frontier 产物现取（reject@0 的两个不同权衡点）
_FR = LAD["frontier"]["reject@0"]["points"]
_FR_DISTINCT: list[dict] = []
for _p in _FR:
    if not _FR_DISTINCT or _FR_DISTINCT[-1]["objectives"] != _p["objectives"]:
        _FR_DISTINCT.append(_p)
for i, _p in enumerate(_FR_DISTINCT[:2], 1):
    _o = _p["objectives"]
    claim("B.2", f"前沿第 {i} 行",
          f"| `{' → '.join(_p['rules'])}` | {_p['drift']:g} | {pct(_o[0])[0]} | {pct(_o[1])[0]} | "
          f"{pct(_o[2])[0]} | {_p['latency_us']:.3f} | {_p['peak_bytes']} |")

# ---------------------------------------------------------------- B.3
OS_ = LIN["order_sensitivity"]
claim("B.3", "无序对总数", f"| {OS_['n_unordered_pairs']} |")
claim("B.3", "敏感对", f"**{OS_['n_order_sensitive_unordered']}**")
claim("B.3", "有序对", f"{OS_['n_ordered_pairs']} 个，其中 {OS_['n_order_sensitive_ordered']} 个不同")
for ex in OS_["examples"][:3]:
    for key in ("values_ab", "values_ba"):
        v = ex[key]
        combos = [f"({a}, {b}, {c})"
                  for a in pct(v[0]) for b in pct(v[1]) for c in pct(v[2])]
        claim("B.3", f"敏感对 {ex['a_then_b'][0]}→{ex['a_then_b'][1]} {key}", *combos)

# ---------------------------------------------------------------- B.4
for name, v in VARIANTS.items():
    if v["kind"] == "illegal" and v["epsilon"] is None:
        claim("B.4", f"{name} 漂移", e4(v["drift"]))
    elif v["epsilon"] is not None:
        claim("B.4", f"{name}({v['epsilon']:g}) 漂移", e4(v["drift"]))
claim("B.4", "漂移系数", f"漂移 ≈ {ADV['drift_per_epsilon']:.6f} · ε")
W = SUM["safe_window"]
claim("B.4", "合法最大漂移", f"**{W['lo']:.15e}**")
claim("B.4", "非法最小漂移", f"**{W['hi']:.15e}**")
claim("B.4", "安全窗口", f"**[{W['lo']:.4e}, {W['hi']:.4e})**，宽 **{W['decades']:.4f}** 个数量级")
claim("B.4", "默认容差", "`1e-9`")
claim("B.4", "默认漏掉", f"**{ADV['n_accepted_at_default_tolerance']}** 个（`{ADV['accepted_at_default_tolerance'][0]}`）")

# ---------------------------------------------------------------- B.5
_seq = [x for i, x in enumerate(MONO_REJECT["n_feasible_space"])
        if i == 0 or x != MONO_REJECT["n_feasible_space"][i - 1]]
claim("B.5", "可行域序列", " → ".join(str(x) for x in _seq))
claim("B.5", "阶梯级数", f"| {len(LAD['rungs']) // 2} |")
claim("B.5", "可行步数", f"{MONO_REJECT['feasible_steps']}")
claim("B.5", "最好均值", f"{SUM['best_answer']['mean_gain']:.12f}")

# ---------------------------------------------------------------- B.6
C = SP["objective_correlation"]
claim("B.6", "speedup~launch", f"**+{C['speedup_gain']['launch_reduction']:.6f}**")
claim("B.6", "speedup~memory", f"-{abs(C['speedup_gain']['memory_saving']):.6f}")
claim("B.6", "launch~memory", f"-{abs(C['launch_reduction']['memory_saving']):.6f}")
claim("B.6", "共线（约数）", f"（+{C['speedup_gain']['launch_reduction']:.4f}）")
claim("B.6", "内存轴 |corr|（约数）", f"|corr| ≈ {abs(C['speedup_gain']['memory_saving']):.2f}")

# ---------------------------------------------------------------- B.7
for name in SR["searchers"]:
    c = R0["cells"][name]
    budget = SR["budget_asymmetry"]["oracle"] if name == "oracle" else SR["budget_per_searcher"]
    claim("B.7", f"{name} 行",
          f"| `{name}` | {budget} | {c['used']} | {c['n_proposed']} | {c['n_unique']} | {c['hits']} | "
          f"{c['n_feasible']} | {c['n_frontier']} | {c['n_frontier_distinct']} | "
          f"{f6(c['hypervolume'])} | {pct(c['hv_gap_vs_oracle'])[0]} |")
claim("B.7", "greedy 缺口", f"缺口 **{pct(CHECK['max_greedy_hv_gap_vs_oracle']['observed'])[0]}**")

# ---------------------------------------------------------------- B.8
for m in SW["margins"]:
    b = m["budget"]
    hv = {n: SW_ROW[b]["cells"][n]["hypervolume"] for n in SW["searchers"]}
    g = m["gain_over_random"]
    claim("B.8", f"预算 {b} 行",
          f"| {b} | {f6(hv['random'])} | {f6(hv['greedy'])} | {f6(hv['beam'])} | {f6(hv['evolution'])} | "
          f"{f6(hv['oracle'])} | {m['best_searcher']} | {'+' if g >= 0 else '-'}{pct(abs(g))[0]} |",
          f"| {b} | {f6(hv['random'])} | {f6(hv['greedy'])} | {f6(hv['beam'])} | {f6(hv['evolution'])} | "
          f"{f6(hv['oracle'])} | {m['best_searcher']} | {pct(abs(g))[0]} |")
claim("B.8", "紧预算", f"| {SW['tight_budget']} |")
claim("B.8", "饱和预算", f"**{SW['random_saturation_budget']}**")

# ---------------------------------------------------------------- B.9
SOFT = EXP_ITEM["soft_constraint_changes_the_answer"]
PEN = EXP_ITEM["penalty_keeps_answers_feasible"]
_SV = "被推翻" if SOFT["verdict"] == "refuted" else "成立"
_PV = "被推翻" if PEN["verdict"] == "refuted" else "成立"
_nd, _nc = SOFT["evidence"]["n_different"], SOFT["evidence"]["n_cells"]
claim("B.9", "格子数", f"**{_nc}**")
claim("B.9", "不同答案的格子", f"| 两种策略给出不同答案的格子 | **{_nd}** |")
claim("B.9", "超体积没变的格子", f"| 其中超体积也变了的格子 | **{sum(1 for e in SOFT['evidence']['examples'] if not e['hv_same'])}** |")
claim("B.9", "penalty 不可行格子", f"| `penalty` 下评分最优候选不可行的格子 | **{PEN['evidence']['n_best_score_infeasible']}** |")
claim("B.9", "penalty 裁定句",
      f"`penalty_keeps_answers_feasible`（原期望 {'True' if PEN['expect'] else 'False'} "
      f"—— 即「软约束会把不可行答案交出去」）",
      f"观测 **{'True' if PEN['observed'] else 'False'}**"
      f"（{PEN['evidence']['n_best_score_infeasible']}/{PEN['evidence']['n_cells']} 个格子的评分最优候选不可行）"
      f"⇒ **{_PV}**")
claim("B.9", "soft 裁定句",
      f"`soft_constraint_changes_the_answer`（原期望 {'True' if SOFT['expect'] else 'False'}）",
      f"观测 **{'True' if SOFT['observed'] else 'False'}**（{_nd}/{_nc} 个格子不同）⇒ **{_SV}**")
claim("B.9", "hv_same 全为 true", f"那 {_nd} 个格子的 `hv_same` 全是 `true`")

# ---------------------------------------------------------------- B.10
for ov, row in COST.items():
    claim("B.10", f"launch_overhead={ov}",
          f"| {ov} | {row['baseline_latency_us']:.3f} | {row['best_latency_us']:.3f} | {row['speedup']:.6f} |",
          f"| {ov}（默认） | {row['baseline_latency_us']:.3f} | {row['best_latency_us']:.3f} | {row['speedup']:.6f} |")
_l0, _l30 = COST[0.0]["speedup"], COST[30.0]["speedup"]
claim("B.10", "launch 贡献", f"**{(_l30 - _l0) / _l30 * 100:.2f}%**")
claim("B.10", "speedup 区间", f"{_l30:.4f} 掉到 {_l0:.4f}")

# ---------------------------------------------------------------- B.11
TE = BM["timing_expectations"][0]
claim("B.11", "样本数", f"| {TE['evidence']['n_plans']} 个前沿计划 |")
claim("B.11", "rho 下界", f"{TE['evidence']['rho']:.2f}")

# ---------------------------------------------------------------- B.12
for name, item in EXP_ITEM.items():
    claim("B.12", f"期望 {name}",
          f"| `{name}` | {'是' if item['expect'] else '否'} | {'是' if item['observed'] else '否'} | "
          f"{'**被推翻**' if item['verdict'] == 'refuted' else '成立'} |")
claim("B.12", "期望汇总",
      f"**{EXP['n_held']} 条成立、{EXP['n_refuted']} 条被推翻、{EXP['n_unmeasurable']} 条算不出来**",
      f"{EXP['n_held']} 成立 / {EXP['n_refuted']} 被推翻")

# ---------------------------------------------------------------- B.13
for name in CHECK:
    claim("B.13", f"门槛 {name}", f"| `{name}`")
claim("B.13", "验收总数", f"{ACC['n_passed']} / {ACC['n']}")
claim("B.13", "门槛条数", f"≥ {CHECK['min_acceptance_checks']['threshold']} | {int(CHECK['min_acceptance_checks']['observed'])}")
claim("B.13", "对抗用例门槛", f"≥ {CHECK['min_adversarial_cases']['threshold']} | {int(CHECK['min_adversarial_cases']['observed'])}")

# ---------------------------------------------------------------- B.15
# 分组条数**从 invariants.py 现数**（README 里那 6/5/14/10/9 不许是凭记忆写的）
INV_SRC = (ROOT / "src" / "graphoptfactory" / "invariants.py").read_text(encoding="utf-8")
_banners = [(m.start(), m.group(1).strip())
            for m in re.finditer(r"^# ={5,} (.+)$", INV_SRC, re.M)]
GROUPS: list[tuple[str, int]] = []
for i, (pos, title) in enumerate(_banners):
    end = _banners[i + 1][0] if i + 1 < len(_banners) else len(INV_SRC)
    GROUPS.append((title, len(re.findall(r"@_case\(", INV_SRC[pos:end]))))
_n_group = sum(n for _, n in GROUPS)
# 硬断言：README 里那张分组表的条数之和，必须**就是**产物里的用例数
assert _n_group == len(ACC["cases"]), f"分组之和 {_n_group} ≠ 用例数 {len(ACC['cases'])}"
claim("B.15", "用例总数", f"**{len(ACC['cases'])} 条，{ACC['n_cases_passed']} 通过，{ACC['n_cases_failed']} 失败**")
for _title, _n in GROUPS:
    claim("B.15", f"分组 {_title}", f"| {_title} | {_n} |")
claim("B.15", "基线行", f"基线：{ACC['n_cases']} 条用例，{ACC['n_cases_passed']} 成立 / {ACC['n_cases_failed']} 不成立")

# 变异测试：**真的跑一遍**，拿它的最后一行（不是抄它文档里的数）
import subprocess  # noqa: E402
_proc = subprocess.run([sys.executable, str(ROOT / ".scratch" / "prove_cases_can_fail.py"),
                        str(OUT)], capture_output=True, text=True, encoding="utf-8")
_summary = next((ln.strip() for ln in reversed(_proc.stdout.splitlines())
                 if "个变异" in ln), "")
_m = re.match(r"(\d+) 个变异，(\d+) 个被抓到，(\d+) 个漏网", _summary)
assert _m, f"变异测试输出无法解析：{_summary!r}\n{_proc.stdout[-2000:]}"
_n_mut, _n_hit, _n_miss = (int(x) for x in _m.groups())
claim("B.15", "变异测试汇总", f"{_n_mut} 个变异，{_n_hit} 个被抓到，{_n_miss} 个漏网")
# 脚本里的 `@mut(...)` 条数必须与它自己报的一致（防「报了数但没跑」）
assert len(re.findall(r"@mut\(", (ROOT / ".scratch" / "prove_cases_can_fail.py")
                      .read_text(encoding="utf-8"))) == _n_mut


# ---------------------------------------------------------------- C.297 / C.300
claim("C.297", "策略差异格子数", f"只有 **{_nd} 个格子**给出的答案不同")
claim("C.297", "hv_same 全为 true", f"那 {_nd} 个的**超体积逐格相同**（`hv_same` 全为 `true`）")
claim("C.297", "soft 裁定", f"`soft_constraint_changes_the_answer` **{_SV}**")
claim("C.300", "抄反的第一行",
      f"| `soft_constraint_changes_the_answer` | 是 / 否 / **被推翻** | 是 / 是 / **成立**"
      f"（{_nd}/{_nc} 个格子不同） |")
claim("C.300", "抄反的第二行",
      f"| `penalty_keeps_answers_feasible` | 否 / 否 / 成立 | 否 / 是 / **被推翻** |")
claim("C.300", "汇总数守恒", f"「{EXP['n_held']} 成立 / {EXP['n_refuted']} 被推翻」")

# 附录 C 的编号区间（结构性检查：从 README 自己的标题数出来）
_c = re.findall(r"^### (\d+)\.", README, re.M)
claim("C", "编号区间", f"本项目占 **{min(_c)}–{max(_c)}**（{len(_c)} 条）")

# 最后一条必须是它自己：断言总数（含本条）—— 放在末尾，`len(claims) + 1` 才对得上。
claims.append(("C.300", "断言条数", (f"{len(claims) + 1} 条断言",)))

# ---------------------------------------------------------------- 比对
misses = [(s, d, ns) for s, d, ns in claims if not any(n in NORM for n in ns)]
print(f"检查 {len(claims)} 条，README 里找不到 {len(misses)} 条：\n")
for s, d, ns in misses:
    print(f"[{s}] {d}\n     期望字符串: {' | '.join(ns)}")
print()

# ---- 反向：README 附录 B 里的浮点 token，哪些在「产物导出的数集」里找不到 ----
B_SEC = NORM.split("## B.1")[1].split("# 附录 C")[0]
exported: set[str] = set()
for _, _, ns in claims:
    for n in ns:
        exported |= set(re.findall(r"\d+\.\d+", n))
# 白名单：这些浮点 token **不是本项目的测量值**，逐条写明理由。
WHITELIST = {
    # 修复前那次运行的读数（参考点按级取时的假象）—— 描述的是一个已不存在的 bug（C.283）
    "0.0321": "参考点按级取时的假象（修复前读数，见 C.283）",
    "0.0540": "同上",
    # 键格式示例，不是测量值（C.306 / B.15：`str(0.0)` → '0.0'）
    "0.0": "`str(0.0)` 的键格式示例，不是测量值",
}
stale = sorted(set(re.findall(r"\d+\.\d+", B_SEC)) - exported - set(WHITELIST))
print(f"附录 B 里出现、但产物导出的数集里没有的浮点 token：{len(stale)} 个"
      f"（白名单 {len(WHITELIST)} 个已排除）")
print(stale)
if stale:
    print("\n白名单里已写明理由的：")
    for t, why in sorted(WHITELIST.items()):
        print(f"  {t}: {why}")
sys.exit(1 if misses or stale else 0)
