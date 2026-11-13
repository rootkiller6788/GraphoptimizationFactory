"""基线图与求值器：构造确定、形状自洽、真的把图算出来。

⚠️ 自包含：不依赖 conftest 夹具。
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from graphoptfactory.config import load_config  # noqa: E402
from graphoptfactory.evaluator import evaluate  # noqa: E402
from graphoptfactory.graph import build_baseline, build_probes, shape_table  # noqa: E402
from graphoptfactory.schema import Graph, GraphError, tensor  # noqa: E402

CONFIG = ROOT / "configs" / "default.json"


def _cfg() -> dict:
    return load_config(CONFIG)


def test_baseline_is_deterministic() -> None:
    a, b = build_baseline(_cfg()), build_baseline(_cfg())
    assert a.digest() == b.digest()
    assert a.to_dict() == b.to_dict()


def test_baseline_shape_table_covers_every_node() -> None:
    cfg = _cfg()
    g = build_baseline(cfg)
    shapes = shape_table(g, cfg)
    assert len(g.nodes) == 49
    assert set(shapes) == set(g.nodes)
    assert shapes[g.outputs[0]] == (cfg["graph"]["rows"], cfg["graph"]["dim"])


def test_evaluator_really_computes_output() -> None:
    cfg = _cfg()
    g = build_baseline(cfg)
    feed = build_probes(cfg)[0]
    out1 = evaluate(g, feed)[g.outputs[0]]
    out2 = evaluate(g, feed)[g.outputs[0]]
    assert out1 == out2
    assert (len(out1), len(out1[0])) == (cfg["graph"]["rows"], cfg["graph"]["dim"])


def test_probes_depend_only_on_probe_params() -> None:
    a = _cfg()
    b = copy.deepcopy(a)
    b["search"]["budget_per_searcher"] += 1
    assert build_probes(a) == build_probes(b)


def test_evaluator_rejects_inner_dimension_mismatch() -> None:
    g = Graph()
    x = g.add("input")
    c = g.add("const", const=tensor([[1.0, 2.0, 3.0]]))
    mm = g.add("matmul", (x, c))
    g.set_outputs((mm,))
    with pytest.raises(GraphError):
        evaluate(g, tensor([[1.0, 2.0]]))
