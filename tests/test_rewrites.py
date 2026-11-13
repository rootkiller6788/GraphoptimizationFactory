"""重写族：注册表自洽、每条合法规则都有靶子、非法对照真的改变语义。

⚠️ 自包含：不依赖 conftest 夹具。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from graphoptfactory.config import load_config  # noqa: E402
from graphoptfactory.graph import build_baseline, build_probes  # noqa: E402
from graphoptfactory.rewrites import (  # noqa: E402
    CONTROL_NAMES,
    CONTROL_RULES,
    LEGAL_NAMES,
    LEGAL_RULES,
    RULES,
    apply_rule,
)
from graphoptfactory.schema import GraphError  # noqa: E402
from graphoptfactory.semantics import is_bit_identical, validate  # noqa: E402

CONFIG = ROOT / "configs" / "default.json"


def _cfg() -> dict:
    return load_config(CONFIG)


def test_registry_is_closed_and_typed() -> None:
    assert len(LEGAL_RULES) == 8
    assert len(CONTROL_RULES) == 2
    assert all(r.kind == "legal" for r in LEGAL_RULES)
    assert all(r.kind == "illegal" for r in CONTROL_RULES)
    assert set(RULES) == set(LEGAL_NAMES) | set(CONTROL_NAMES)


def test_every_legal_rule_has_a_target_in_baseline() -> None:
    cfg = _cfg()
    baseline = build_baseline(cfg)
    missed = [name for name in LEGAL_NAMES
              if apply_rule(baseline, name, cfg)[1] <= 0]
    assert not missed, f"基线图里没有靶子的合法规则：{missed}"


def test_illegal_controls_hit_and_change_semantics() -> None:
    cfg = _cfg()
    baseline = build_baseline(cfg)
    probes = build_probes(cfg)
    for name in CONTROL_NAMES:
        candidate, hits = apply_rule(baseline, name, cfg)
        assert hits > 0, f"{name} 在基线图上没有命中"
        result = validate(baseline, candidate, probes, 0.0)
        assert not result.ok, f"{name} 应该改变语义"
        assert result.drift > 1e-6, f"{name} 的漂移太小：{result.drift}"


def test_exactly_five_legal_rules_are_bit_identical() -> None:
    cfg = _cfg()
    baseline = build_baseline(cfg)
    probes = build_probes(cfg)
    exact = [name for name in LEGAL_NAMES
             if is_bit_identical(baseline, apply_rule(baseline, name, cfg)[0], probes)]
    assert len(exact) == 5, f"逐位等价的合法规则应为 5 条，实得 {exact}"
    assert set(LEGAL_NAMES) - set(exact) == {
        "reassociate_add", "matmul_reassociate", "scale_into_weight"}


def test_unknown_rule_raises() -> None:
    with pytest.raises(GraphError):
        apply_rule(build_baseline(_cfg()), "no_such_rule", _cfg())
