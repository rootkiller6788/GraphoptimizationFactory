# 变更日志

格式参照 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循语义化版本。本项目产物另带内容寻址的 `v1-<digest>` 版本（见 `out/version.json`）。

## [Unreleased]

### Added

- `python -m graphoptfactory` 模块入口，与 `main.py` 共用 `cli.main`。
- CLI `--version`，版本号唯一来源是 `graphoptfactory.__version__`。
- `tests/` 下的确定性随机、配置、基线图 / 求值器、重写族测试。
- 本变更日志。

### Fixed

- 非 UTF-8 控制台（Windows GBK / CI 管道）打印 `µ`、`τ` 时抛
  `UnicodeEncodeError`，导致 `check` / `show` 失败；入口统一把
  stdout/stderr 切到 UTF-8。
- `tests/conftest.py` 引用了不存在的 `diagnostics/` 目录。
- `rng.py` 文档声称 `Rng` 会先排序再抽取，与实现不符。

## [1.0.0] - 初始提交

### Added

- 世界与求值器：基线图（49 节点）、形状推导、纯 Python 求值（`math.sumprod`）、
  三目标成本模型。
- 重写族：8 条合法重写（5 条逐位等价 / 3 条近似等价）+ 2 条非法对照，
  以及 `bias_epsilon(ε)` 五档扰动。
- 搜索与前沿：oracle / random / greedy / beam / evolution 五个搜索器、
  两层缓存、Pareto 工具、7 档预算扫描。
- 交付：τ × 策略阶梯（10 × 2）、对抗注入、17 条设计期望裁定、
  19 条验收门槛、44 条对抗用例、报告渲染与只读重建。
- 运行命令：`run` / `check` / `visualize` / `verify` / `show`。
