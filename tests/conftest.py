"""pytest 公共夹具：把 `src/` 放进 `sys.path`。

⚠️ 这个 `tests/` 是**入口**，不是第二套判据 —— 判据的唯一真源是流水线自身的
运行时核对（`acceptance.py` 的 19 条门槛 + `invariants.py` 的 44 条对抗用例）。
这里只放冒烟与契约测试：`test_import` 是接线闸，`test_cli_contract` 钉住 CLI 行为。
⚠️ `test_import.py` / `test_cli_contract.py` **自包含**，不依赖这里的夹具 ——
因为有的项目自带别的 `conftest.py`，依赖夹具会因「夹具不存在」假红。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
