"""配置：默认值自洽、深合并语义、陈旧键保险丝、协议指纹。

⚠️ 自包含：不依赖 conftest 夹具。
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from graphoptfactory.config import (  # noqa: E402
    DEFAULTS,
    ConfigError,
    deep_merge,
    load_config,
    no_stale_config_keys,
    protocol_digest,
    validate_config,
)
from graphoptfactory.rewrites import LEGAL_NAMES  # noqa: E402


def test_default_config_is_valid() -> None:
    validate_config(copy.deepcopy(DEFAULTS))


def test_checked_in_config_loads_without_stale_keys() -> None:
    path = ROOT / "configs" / "default.json"
    disk = json.loads(path.read_text(encoding="utf-8"))
    assert no_stale_config_keys(disk, DEFAULTS) == []
    cfg = load_config(path)
    assert cfg["seed"] == DEFAULTS["seed"]


def test_deep_merge_merges_dicts_but_replaces_lists() -> None:
    base = {"a": {"x": 1, "y": 2}, "items": [1, 2]}
    override = {"a": {"y": 9}, "items": [3]}
    out = deep_merge(base, override)
    assert out == {"a": {"x": 1, "y": 9}, "items": [3]}
    assert base == {"a": {"x": 1, "y": 2}, "items": [1, 2]}


def test_stale_keys_are_reported_recursively() -> None:
    disk = {"graph": {"n_layers": 1, "old": 2}, "old_top": 1}
    assert no_stale_config_keys(disk, DEFAULTS) == ["graph.old", "old_top"]


def test_protocol_digest_ignores_path_keys() -> None:
    cfg = copy.deepcopy(DEFAULTS)
    a = protocol_digest({**cfg, "_config_path": "a.json", "_root": "x"})
    b = protocol_digest({**cfg, "_config_path": "b.json", "_root": "y"})
    assert a == b


def test_validate_rejects_depth_above_rule_count() -> None:
    cfg = copy.deepcopy(DEFAULTS)
    cfg["search"]["max_depth"] = len(LEGAL_NAMES) + 1
    with pytest.raises(ConfigError):
        validate_config(cfg)
