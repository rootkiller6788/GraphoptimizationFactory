"""对抗用例的**对抗用例**：证明它们真的会红。

一条从不触发的用例与没有用例在输出里长得一模一样。
做法：把产物按 `finish()` 的内存形状装好，然后**逐个注入故障**，
断言「该红的那条用例红了，而且只有它红」。

用法：python .scratch/prove_cases_can_fail.py <out_dir>
"""
from __future__ import annotations

import copy
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from graphoptfactory.invariants import run_cases  # noqa: E402

NAMES = ("config.json", "baseline.json", "space.json", "lineage.json",
         "adversarial.json", "search.json", "budget_sweep.json", "ladder.json",
         "benchmark.json", "summary.json", "expectations.json",
         "acceptance.json")


def load(out_dir: pathlib.Path) -> dict:
    return {n: json.loads((out_dir / n).read_text(encoding="utf-8"))
            for n in NAMES}


def rung(p, strategy, tol):
    return next(r for r in p["ladder.json"]["rungs"]
                if r["strategy"] == strategy and r["tolerance"] == tol)


def cells(p):
    for r in p["ladder.json"]["rungs"]:
        for n in sorted(r["cells"]):
            yield r, r["cells"][n]


# 每个变异：名字 → (改法, 期望变红的用例名)
MUTATIONS: list[tuple[str, object, str]] = []


def mut(name, target):
    def deco(fn):
        MUTATIONS.append((name, fn, target))
        return fn
    return deco


@mut("空间大小 +1", "space_size_matches_the_permutation_count")
def _m1(p):
    p["space.json"]["space_size"] += 1


@mut("某条规则的 hits 清零", "every_rule_fires_at_least_once")
def _m2(p):
    p["lineage.json"]["rules"][0]["hits"] = 0


@mut("把一条合法规则标成逐位等价但漂移非 0",
     "rule_drift_and_bit_identity_agree")
def _m3(p):
    for r in p["lineage.json"]["rules"]:
        if r["kind"] == "legal" and not r["bit_identical"]:
            r["bit_identical"] = True
            break


@mut("把裁定改反（不改读数）",
     "expectation_verdicts_match_expect_and_observed")
def _m4(p):
    it = p["expectations.json"]["items"][0]
    it["verdict"] = "refuted" if it["verdict"] == "held" else "held"


@mut("把时序期望塞回版本化产物",
     "timing_expectations_are_out_of_the_versioned_products")
def _m5(p):
    p["expectations.json"]["items"][0]["timing_dependent"] = True


@mut("超体积缺口列改成另一个口径",
     "hv_gap_column_matches_the_hypervolumes")
def _m6(p):
    for _, c in cells(p):
        c["hv_gap_vs_oracle"] = float(c["hv_gap_vs_oracle"]) + 0.01


@mut("某格子的 used 超过预算", "searchers_respect_their_budget")
def _m7(p):
    for _, c in cells(p):
        if c["searcher"] != "oracle":
            c["used"] = int(c["budget"]) + 1
            break


@mut("n_feasible 超过可行域", "feasible_count_never_exceeds_the_feasible_space")
def _m8(p):
    for r, c in cells(p):
        if r["strategy"] == "reject" and r["tolerance"] == 0.0:
            c["n_feasible"] = int(r["n_feasible_space"]) + 7
            break


@mut("提案记账对不上（hits 改成 0）", "proposal_accounting_adds_up")
def _m9(p):
    for _, c in cells(p):
        if int(c["hits"]) > 0:
            c["hits"] = 0
            break


@mut("前沿里混进一个被支配的点",
     "frontier_members_are_pairwise_nondominated")
def _m10(p):
    """⚠️ 第一版是「复制 frontier[0] 追加」—— 那是**变异写错了**：
    重复点在支配关系下互不支配（`dominates` 要求至少一个目标严格更差），
    所以用例不红是对的，不是漏网。
    要真的考用例，必须注入一个**被别的成员支配**的点：
    取三个目标各自的最小值（这个点被前沿里**每一个**成员支配），
    复制一个现有 trial 的字段（保住 drift/feasible），只改 objectives 与 plan_id。
    """
    for _, c in cells(p):
        if len(c["frontier"]) >= 2:
            mins = [min(float(t["objectives"][i]) for t in c["frontier"])
                    for i in range(3)]
            bad = copy.deepcopy(c["frontier"][0])
            bad["plan_id"] = "mutant-dominated"
            bad["objectives"] = mins
            c["frontier"].append(bad)
            c["n_frontier"] = len(c["frontier"])
            c["n_frontier_distinct"] = len(
                {tuple(round(x, 12) for x in t["objectives"])
                 for t in c["frontier"]})
            break


@mut("前沿成员的漂移超过本级容差",
     "frontier_members_drift_within_the_rung_tolerance")
def _m11(p):
    r = rung(p, "reject", 0.0)
    r["cells"]["oracle"]["frontier"][0]["drift"] = 0.5


@mut("参考点改成不被支配的", "hypervolume_reference_point_is_dominated")
def _m12(p):
    p["space.json"]["reference"] = [0.9, 0.9, 0.9]


@mut("摘要与明细分叉", "summary_is_a_projection_of_other_products")
def _m13(p):
    p["summary.json"]["space_size"] = 999


@mut("删掉一条登记（模拟加了新产物忘登记）",
     "versioned_and_unversioned_cover_every_product")
def _m14(p):
    p["brand_new_product.json"] = {"x": 1}


@mut("安全窗口端点改成 ε 网格里的数", "safe_window_matches_the_raw_drifts")
def _m15(p):
    p["expectations.json"]["derived"]["safe_window"]["hi"] = 1e-9


@mut("成本模型的基线延迟对 launch 开销方向反了",
     "cost_model_is_monotone_in_launch_overhead")
def _m16(p):
    """⚠️ 第一版是「把 cost_sweep 列表反转」—— 那是**变异写错了**：
    用例内部先按 `launch_overhead_us` 排序，反转列表对它**恒等**。
    要真的考用例，必须破坏**单调性本身**：按开销排好序后，
    把 `baseline_latency_us` 这一列**倒着填回去** ——
    原序列严格递增，倒填即严格递减 ⇒ 第一对就违反 `b >= a`。
    """
    cs = sorted(p["benchmark.json"]["cost_sweep"],
                key=lambda c: float(c["launch_overhead_us"]))
    vals = [float(c["baseline_latency_us"]) for c in cs][::-1]
    for c, v in zip(cs, vals):
        c["baseline_latency_us"] = v


@mut("把一条非法重写在窗口内放过去",
     "illegal_rewrites_are_all_rejected_below_the_window")
def _m17(p):
    for v in p["adversarial.json"]["variants"]:
        if v.get("epsilon") is not None:
            v["accepted_at"]["1e-15"] = True
            break


@mut("把 ε 网格声明改成全在一侧", "epsilon_grid_spans_the_cliff")
def _m18(p):
    p["adversarial.json"]["eps_rows_accepted"] = 0


@mut("配置里的规则数与血缘表不一致",
     "rule_set_is_partitioned_into_legal_and_control")
def _m19(p):
    p["lineage.json"]["rules"] = [r for r in p["lineage.json"]["rules"]
                                  if r["rule"] != "operator_fusion"]


@mut("算子计数之和对不上节点数", "op_counts_sum_to_the_node_count")
def _m20(p):
    p["baseline.json"]["op_counts"]["add"] += 1


@mut("τ=0 那一级放过一个非法重写（守住 accepted_at 的键口径）",
     "illegal_rewrites_are_all_rejected_below_the_window")
def _m21(p):
    """这个变异**只**在键口径正确时才会被抓到。

    `adversarial.py` 用 `str(t)` 造键（`0.0 → '0.0'`）。
    如果用例用 `f"{t:g}"` 去查（`0.0 → '0'`），τ=0 那一格永远查不到 ⇒ 漏网。
    ⇒ 这条变异是那个 bug 的**回归守卫**。
    """
    tol0 = min(float(x) for x in
               p["config.json"]["resolved"]["ladder"]["tolerances"])
    v = next(v for v in p["adversarial.json"]["variants"]
             if v.get("epsilon") is not None)
    v["accepted_at"][str(tol0)] = True


def main() -> int:
    out_dir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                           else ROOT / "out")
    base = load(out_dir)
    base_res = run_cases(copy.deepcopy(base))
    print(f"基线：{base_res['n_cases']} 条用例，"
          f"{base_res['n_cases_passed']} 成立 / {base_res['n_cases_failed']} 不成立")
    if base_res["n_cases_failed"]:
        print("基线就不绿，先修基线：", base_res["failed_cases"])
        return 1

    print()
    bad = 0
    for name, fn, target in MUTATIONS:
        p = copy.deepcopy(base)
        fn(p)
        res = run_cases(p)
        failed = set(res["failed_cases"])
        hit = target in failed
        extra = sorted(failed - {target})
        mark = "OK  " if hit and not extra else ("OK* " if hit else "MISS")
        if not hit:
            bad += 1
        print(f"{mark} {name}")
        print(f"      期望红：{target}")
        print(f"      实际红：{sorted(failed) if failed else '（无）'}")
        if extra:
            print(f"      ⚠️ 附带红的（可能是用例之间有耦合）：{extra}")
    print()
    print(f"{len(MUTATIONS)} 个变异，{len(MUTATIONS) - bad} 个被抓到，{bad} 个漏网")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
