"""对抗：注入**已知幅度**的非法重写，看校验器在哪一档容差把它们收下。

## 为什么用「已知幅度」的注入而不是随机破坏

随机破坏只能得到「校验器漏了一些」这种没有量纲的结论。
本项目的注入是**可解析的**：

- `relu_to_identity`：把 relu 换掉，负值不再截断 —— 漂移是 O(1) 量级；
- `drop_bias`：丢掉 bias —— 漂移也是 O(1)；
- `bias_epsilon(ε)`：给 bias 加一个 **ε 量级**的扰动。
  这一条是标尺：**漂移 ≈ 1.519·ε**（1.519 是实测出来的系数，不是设定的）。

于是「容差该取多少」从一个审美问题变成一个**可测量的数**：
容差崖 = 合法重写的最大漂移（≈1.5e-16）与非法重写的最小漂移（≈1.5e-12）
之间那四个数量级。

## 这一节最重要的产出是一条**否定性**结论

`semantics.tolerance` 的默认值是 `1e-9`。而 `bias_epsilon(1e-12)` 的漂移是
`1.5e-12 < 1e-9` ⇒ **它在默认容差下被接受**。
一个 1e-12 的扰动被放过去了，而校验器会报「通过」。

这不是 bug，是**配置取值的后果** —— 也正是要测的东西：
「校验器能不能挡住非法重写」这个问题的答案取决于容差取多少，
而容差是配置里的一个数。**一个不看容差的「校验器通过率」是没有意义的读数。**

## ε 网格必须横跨崖的两侧（这条断言只能在这里做）

配置校验只看得到 ε 与 τ 两个数、量不出漂移，所以它**不断言**两者的大小关系
（在 `config.py` 里写明了原因）。真正的断言在这里：漂移算出来之后，
网格必须在「全部被接受」与「全部被拒」之间至少跨过一次，
否则「崖在哪」这个数根本没被测量到 —— 而它不会报错，只会给出一张
看起来很正常、实际上每一格都一样的表。
"""

from __future__ import annotations

from typing import Any, Sequence

from .cost import metrics
from .graph import shape_table
from .objective import vector
from .rewrites import bias_epsilon, drop_bias, relu_to_identity
from .schema import Graph, Tensor
from .semantics import output_drift, run_outputs


def _drift_of(baseline: Graph, variant: Graph, probes: Sequence[Tensor]) -> float:
    worst = 0.0
    for feed in probes:
        worst = max(worst, output_drift(run_outputs(baseline, feed),
                                        run_outputs(variant, feed)))
    return worst


def run_adversarial(cfg: dict[str, Any], baseline: Graph,
                    probes: Sequence[Tensor],
                    tolerances: Sequence[float]) -> dict[str, Any]:
    base_m = metrics(baseline, shape_table(baseline, cfg), cfg)

    variants: list[dict[str, Any]] = []
    for name, fn in (("relu_to_identity", relu_to_identity),
                     ("drop_bias", drop_bias)):
        g, hits = fn(baseline, cfg)
        d = _drift_of(baseline, g, probes)
        variants.append({
            "variant": name, "kind": "illegal", "epsilon": None,
            "hits": hits, "drift": d,
            "accepted_at": {str(t): bool(d <= t) for t in tolerances},
        })

    eps_rows: list[dict[str, Any]] = []
    for eps in [float(x) for x in cfg["adversarial"]["bias_epsilons"]]:
        g, hits = bias_epsilon(baseline, cfg, eps)
        d = _drift_of(baseline, g, probes)
        row = {
            "variant": f"bias_epsilon({eps:g})", "kind": "illegal",
            "epsilon": eps, "hits": hits, "drift": d,
            "drift_per_epsilon": (d / eps if eps else float("nan")),
            "accepted_at": {str(t): bool(d <= t) for t in tolerances},
        }
        variants.append(row)
        eps_rows.append(row)

    # ---- 崖：每个变体在**最小**哪一级容差被接受
    for v in variants:
        accepted = [t for t in tolerances if v["drift"] <= t]
        v["cliff_tolerance"] = min(accepted) if accepted else None
        v["accepted_any"] = bool(accepted)

    # ---- 断言：ε 网格必须横跨崖的两侧（见模块文档）
    n_eps = len(eps_rows)
    n_acc = sum(1 for r in eps_rows if r["accepted_any"]) if eps_rows else 0
    if eps_rows:
        if n_acc == 0 or n_acc == n_eps:
            raise RuntimeError(
                "ε 网格全在崖的同一侧："
                f"{n_acc}/{n_eps} 档在某一级容差下被接受。"
                "「崖在哪」这个数没有被测量到 —— 请调整 "
                "adversarial.bias_epsilons 或 ladder.tolerances。"
            )
    # ⚠️ 这一条**必须是算出来的**：第一版写死成 `True`，于是「ε 网格横跨崖的两侧」
    #    这句话在产物里**永远成立** —— 它看起来像一个测量结果，其实是一个假设。
    #    上面的 RuntimeError 挡住了流水线，但产物里那个字段是给读者看的，
    #    读者没法从产物内部发现它是个常量。
    spans = bool(eps_rows) and 0 < n_acc < n_eps

    # ---- 逐级：非法变体被接受的个数（0 是设计目标，但**它依赖容差取值**）
    by_tol: list[dict[str, Any]] = []
    for t in tolerances:
        acc = [v["variant"] for v in variants if v["accepted_at"][str(t)]]
        by_tol.append({
            "tolerance": t, "n_illegal_accepted": len(acc), "accepted": acc,
        })

    # ---- 默认容差下漏掉了几个（**这是本项目最想报的一个数**）
    default_tol = float(cfg["semantics"]["tolerance"])
    default_acc = [v["variant"] for v in variants
                   if v["accepted_at"][str(default_tol)]]

    return {
        "baseline_metrics": base_m,
        "variants": variants,
        "by_tolerance": by_tol,
        "default_tolerance": default_tol,
        "n_illegal_variants": len(variants),
        "n_accepted_at_default_tolerance": len(default_acc),
        "accepted_at_default_tolerance": default_acc,
        "drift_per_epsilon": (round(sum(r["drift_per_epsilon"] for r in eps_rows)
                                    / len(eps_rows), 12) if eps_rows else None),
        #: **算出来的**（不是写死的）：ε 网格里有多少档在某一级容差下被接受。
        #: 0 或全部 ⇒ 网格全在崖的一侧 ⇒「崖在哪」没被测量到。
        "eps_grid_spans_cliff": spans,
        "eps_rows_accepted": n_acc,
        "eps_rows_total": n_eps,
        "objective_deltas": _objective_deltas(baseline, cfg, base_m, probes),
    }


def _objective_deltas(baseline: Graph, cfg: dict[str, Any],
                      base_m: dict[str, Any],
                      probes: Sequence[Tensor]) -> list[dict[str, Any]]:
    """非法变体的**目标读数**：它们看起来有多好。

    一个只报漂移的对抗表是不完整的：漂移 1.13 的 `relu_to_identity`
    同时把延迟降了多少？如果它降得**比合法重写还多**，
    那么「不放松容差」这件事就有了代价 —— 而这个代价必须被报出来，
    否则「安全窗口」会被读成一个纯收益。
    """
    out: list[dict[str, Any]] = []
    for name, fn in (("relu_to_identity", relu_to_identity),
                     ("drop_bias", drop_bias)):
        g, _ = fn(baseline, cfg)
        m = metrics(g, shape_table(g, cfg), cfg)
        out.append({"variant": name, "kind": "illegal",
                    "objectives": [round(x, 12) for x in vector(base_m, m)],
                    "metrics": {k: m[k] for k in sorted(m)}})
    return out
