"""`python -m graphoptfactory` 入口。

与 `main.py` 等价（报告 / CLI 行为完全一致），面向已把 `src/` 放进
`sys.path`（或已 `pip install -e .`）的使用者。
"""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
