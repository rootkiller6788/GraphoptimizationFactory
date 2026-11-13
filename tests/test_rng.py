"""确定性随机：种子派生稳定、同种子逐位复现、空序列不静默。

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

from graphoptfactory.rng import Rng, derive_seed, digest_of, sorted_unique  # noqa: E402


def test_derive_seed_is_stable_and_sensitive() -> None:
    parts = (20261003, "weights", 3, 4, 8)
    assert derive_seed(*parts) == derive_seed(*parts)
    assert derive_seed(1, "a") != derive_seed(1, "b")
    assert 0 <= derive_seed("x") < 2**64


def test_same_seed_gives_identical_stream() -> None:
    a, b = Rng(42), Rng(42)
    assert [a.random() for _ in range(8)] == [b.random() for _ in range(8)]
    assert a.randint(0, 99) == b.randint(0, 99)


def test_choice_rejects_empty_sequence() -> None:
    with pytest.raises(ValueError):
        Rng(1).choice([])


def test_sorted_unique_is_order_independent() -> None:
    assert sorted_unique(["b", "a", "b"]) == ["a", "b"]
    assert sorted_unique([3, 1, 2, 1]) == [1, 2, 3]


def test_digest_of_is_stable_and_short() -> None:
    assert digest_of("graph", 1) == digest_of("graph", 1)
    assert len(digest_of("graph", 1)) == 16
