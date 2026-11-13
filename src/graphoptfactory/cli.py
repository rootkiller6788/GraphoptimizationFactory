"""命令行。

## 路径锚定契约

`--config` / `--out` / `--root` 的**相对路径按项目根解析**，不按当前工作目录。
`_anchor_args` 在 `main()` 里紧跟 `parse_args` 锚一次 ——
不锚的话，「从工作区根调用」与「从项目目录调用」会写到两个不同的地方，
而两者都不报错。

## `--json` 的纯度

`--json` 时 stdout 上**只许有一段 JSON**（不许先打印说明再追加 JSON，那样没法 `| jq`）。
手法：`_emit_json` 走真正的 stdout，命令里的 `print` 被 `_quiet_prose` 上下文吞掉。

## 只读命令遇缺失产物

在内存里重跑一遍，**绝不落盘**，并明说「本次是内存重跑，未写盘」。
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator

from . import pipeline
from .config import load_config
from .emit import emit
from .report import render_html, render_markdown

ROOT = Path(__file__).resolve().parents[2]     # L6-graphoptimizationfactory/

_JSON_OUT = sys.stdout                          # 真正的 stdout（供 --json 用）


@contextlib.contextmanager
def _quiet_prose(enabled: bool) -> Iterator[None]:
    """`--json` 时把命令里的 `print` 吞掉，保证 stdout 上只有一段 JSON。"""
    if not enabled:
        yield
        return
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield


def _emit_json(obj: Any) -> None:
    _JSON_OUT.write(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    _JSON_OUT.flush()


def _anchor_args(args: argparse.Namespace) -> argparse.Namespace:
    """把相对路径锚到项目根。**紧跟 parse_args 调一次。**"""
    root = Path(getattr(args, "root", None) or ROOT).resolve()
    for name in ("config", "out", "target"):
        v = getattr(args, name, None)
        if v is None:
            continue
        p = Path(v)
        setattr(args, name, p if p.is_absolute() else (root / p))
    args.root = root
    return args


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="graphoptfactory",
        description="GraphOptimizationFactory：重写计划的搜索闭环（硬约束 = 语义等价）",
    )
    p.add_argument("--root", default=None, help="项目根（默认 = 本文件所在项目）")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        # ⚠️ 默认值必须用 SUPPRESS：否则子解析器会把主解析器已设好的
        #    `--json` 覆写回 False（子解析器的默认值优先级更高）。
        sp.add_argument("--config", default=argparse.SUPPRESS, help="配置 JSON")
        sp.add_argument("--out", default=argparse.SUPPRESS, help="产物目录")
        sp.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                        help="stdout 上只输出一段 JSON")

    run = sub.add_parser("run", help="跑完整流水线并落盘")
    common(run)

    vis = sub.add_parser("visualize", help="从 out/ 重建报告（只读）")
    common(vis)
    vis.add_argument("--target", default=argparse.SUPPRESS,
                     help="需要落盘时写到这个目录（默认不写盘）")

    chk = sub.add_parser("check", help="从 out/ 重跑验收（只读）")
    common(chk)

    ver = sub.add_parser("verify", help="确定性与产物充分性核对")
    common(ver)

    sh = sub.add_parser("show", help="打印摘要")
    common(sh)

    return p


# --------------------------------------------------------------------- 子命令
def _resolve(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    cfg = load_config(getattr(args, "config", None), root=args.root)
    out = Path(getattr(args, "out", None) or (args.root / "out"))
    return cfg, out


def cmd_run(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))
    with _quiet_prose(json_mode):
        cfg, out = _resolve(args)
        run = pipeline.build(cfg)
        pipeline.finish(run)
        manifest = emit(
            out, run.products, pipeline.VERSIONED_FILES, pipeline.UNVERSIONED_FILES,
            extra=lambda p: {"report.md": render_markdown(p),
                             "report.html": render_html(p)},
        )
        run.timings["total"] = run.timings.get("total", 0.0)
        summary = run.products["summary.json"]
        text = (f"版本 {manifest['version']}\n"
                f"空间 {summary['space_size']} 个计划，"
                f"{summary['n_distinct_digests']} 张不同的图\n"
                f"τ=0 前沿 {summary['frontier_points_at_tau0']} 点，"
                f"超体积 {summary['hypervolume_at_tau0']}\n"
                f"期望：{summary['expectations']['n_held']} 成立 / "
                f"{summary['expectations']['n_refuted']} 被推翻 / "
                f"{run.products['expectations.json']['n_unmeasurable']} 算不出来\n"
                f"验收：{summary['acceptance']['n_passed']}/"
                f"{summary['acceptance']['n']} 通过\n"
                f"产物目录：{out}")
        if not json_mode:
            print(text)
    if json_mode:
        _emit_json({"cmd": "run", "out": str(out),
                    "version": manifest["version"],
                    "summary": run.products["summary.json"],
                    "acceptance_failed": run.products["acceptance.json"]["failed"],
                    "refuted": [i["name"] for i in run.products["expectations.json"]["items"]
                                if i["verdict"] == "refuted"]})
    return 0 if not run.products["acceptance.json"]["failed"] else 0


def _products_for_readonly(args: argparse.Namespace) -> tuple[dict[str, Any], bool, str]:
    """读 `out/`；缺产物则在内存里重跑（**绝不落盘**）。"""
    cfg, out = _resolve(args)
    from .emit import read_products
    from .visualize import REQUIRED
    missing = [n for n in REQUIRED if not (out / n).exists()]
    if not missing:
        return read_products(out, REQUIRED), False, ""
    run = pipeline.build(cfg)
    pipeline.finish(run)
    # 报告渲染要读 version.json —— 内存重跑时补一个**不落盘**的版本清单
    if "version.json" not in run.products:
        run.products["version.json"] = {
            "version": "(内存重跑，未写盘)",
            "versioned_files": list(pipeline.VERSIONED_FILES),
            "unversioned_files": list(pipeline.UNVERSIONED_FILES),
            "digests": {}, "created_at": None,
            "note": "本次是内存重跑，未写盘。",
        }
    return run.products, True, f"缺产物 {missing}，本次是内存重跑，未写盘"


def cmd_visualize(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))
    with _quiet_prose(json_mode):
        cfg, out = _resolve(args)
        products, reran, note = _products_for_readonly(args)
        md = render_markdown(products)
        html = render_html(products)
        target = getattr(args, "target", None)
        if target is not None:
            Path(target).mkdir(parents=True, exist_ok=True)
            (Path(target) / "report.md").write_text(md, encoding="utf-8", newline="\n")
            (Path(target) / "report.html").write_text(html, encoding="utf-8",
                                                      newline="\n")
            written = [str(Path(target) / "report.md"),
                       str(Path(target) / "report.html")]
        else:
            written = []
        rows = []
        for name, text in (("report.md", md), ("report.html", html)):
            p = out / name
            existing = p.read_text(encoding="utf-8") if p.exists() else None
            rows.append({"file": name, "exists": existing is not None,
                         "identical": existing == text,
                         "rebuilt_bytes": len(text.encode("utf-8"))})
        result = {
            "cmd": "visualize", "out": str(out), "read_only": not written,
            "in_memory_rerun": reran, "note": note or "只读：未写盘。",
            "files": rows, "all_identical": all(r["identical"] for r in rows),
            "written": written,
        }
        if not json_mode:
            print(f"out/ 报告与重建结果："
                  f"{'逐位相同' if result['all_identical'] else '**不一致**'}")
            for r in rows:
                print(f"  {r['file']:12s} 存在={r['exists']} "
                      f"相同={r['identical']} 重建 {r['rebuilt_bytes']} 字节")
            print(result["note"])
            for w in written:
                print(f"  写出 {w}")
    if json_mode:
        _emit_json(result)
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))
    with _quiet_prose(json_mode):
        cfg, out = _resolve(args)
        products, reran, note = _products_for_readonly(args)
        acc = products["acceptance.json"]
        exp = products["expectations.json"]
        if not json_mode:
            print(f"验收 {acc['n_passed']}/{acc['n']} 通过")
            for c in acc["checks"]:
                mark = "OK  " if c["passed"] else "FAIL"
                print(f"  [{mark}] {c['name']:38s} {c['detail']}")
            print(f"期望：{exp['n_held']} 成立 / {exp['n_refuted']} 被推翻 / "
                  f"{exp['n_unmeasurable']} 算不出来")
            for i in exp["items"]:
                if i["verdict"] != "held":
                    print(f"  [{i['verdict']:12s}] {i['name']}: {i['detail']}")
            print(note or "只读：未写盘。")
    if json_mode:
        _emit_json({"cmd": "check", "out": str(out), "in_memory_rerun": reran,
                    "note": note or "只读：未写盘。",
                    "acceptance": acc, "expectations": exp})
    return 0 if not acc["failed"] else 1


def cmd_verify(args: argparse.Namespace) -> int:
    """确定性 + 产物充分性。**两次 run 只许在时间戳与耗时上不同。**"""
    json_mode = bool(getattr(args, "json", False))
    cfg, _out = _resolve(args)
    with _quiet_prose(json_mode):
        tmp = Path(tempfile.mkdtemp(prefix="graphopt-verify-"))
        try:
            versions = []
            dirs = []
            for i in (1, 2):
                d = tmp / f"run{i}"
                run = pipeline.build(cfg)
                pipeline.finish(run)
                manifest = emit(
                    d, run.products, pipeline.VERSIONED_FILES,
                    pipeline.UNVERSIONED_FILES,
                    extra=lambda p: {"report.md": render_markdown(p),
                                     "report.html": render_html(p)},
                )
                versions.append(manifest["version"])
                dirs.append(d)

            all_files = sorted({p.name for d in dirs for p in d.iterdir()})
            rows = []
            for name in all_files:
                a = (dirs[0] / name).read_bytes()
                b = (dirs[1] / name).read_bytes()
                rows.append({"file": name, "identical": a == b,
                             "bytes": len(a)})

            # 产物充分性：从产物重建报告 == 直接渲染
            from .emit import read_products
            from .visualize import REQUIRED
            p1 = read_products(dirs[0], REQUIRED)
            suff = {
                "report.md": render_markdown(p1) == (dirs[0] / "report.md")
                .read_text(encoding="utf-8"),
                "report.html": render_html(p1) == (dirs[0] / "report.html")
                .read_text(encoding="utf-8"),
            }

            expect_diff = {"version.json", "benchmark.json",
                           "report.md", "report.html"}
            unexpected = [r["file"] for r in rows
                          if not r["identical"] and r["file"] not in expect_diff]
            result = {
                "cmd": "verify",
                "version_1": versions[0], "version_2": versions[1],
                "same_version": versions[0] == versions[1],
                "files": rows,
                "expected_to_differ": sorted(expect_diff),
                "expected_to_differ_why": {
                    "version.json": "created_at 时间戳（唯一允许印时间戳的地方）",
                    "benchmark.json": "实测耗时",
                    "report.md": "视图，且正文里嵌了耗时表",
                    "report.html": "视图，且正文里嵌了耗时表",
                },
                "unexpectedly_different": unexpected,
                "products_sufficient": suff,
                "products_sufficient_all": all(suff.values()),
                "ok": (versions[0] == versions[1] and not unexpected
                       and all(suff.values())),
            }
            if not json_mode:
                print(f"版本 {versions[0]} / {versions[1]} "
                      f"（相同={result['same_version']}）")
                for r in rows:
                    tag = "逐位相同" if r["identical"] else "**不同**"
                    print(f"  {r['file']:18s} {tag:8s} {r['bytes']} 字节")
                print(f"预期不同的文件：{sorted(expect_diff)}")
                print(f"**意外不同**的文件：{unexpected or '（无）'}")
                print(f"产物充分性（从产物重建报告 == 直接渲染）：{suff}")
                print("结论：" + ("**全部通过**" if result["ok"] else "**有核对未通过**"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    if json_mode:
        _emit_json(result)
    return 0 if result["ok"] else 1


def cmd_show(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))
    with _quiet_prose(json_mode):
        products, reran, note = _products_for_readonly(args)
        s = products["summary.json"]
        if not json_mode:
            print(f"版本 {products['version.json']['version']}")
            print(f"空间 {s['space_size']} 计划 / {s['n_distinct_digests']} 张不同的图")
            print(f"基线 {s['baseline']['nodes']} 节点、"
                  f"{s['baseline']['launches']} launches、"
                  f"{s['baseline']['latency_us']} µs")
            best = s.get("best_answer") or {}
            if best:
                print(f"最好答案 {' → '.join(best['rules'])}："
                      f"延迟 {best['objectives'][0]:.2%} / "
                      f"内存 {best['objectives'][1]:.2%} / "
                      f"launch {best['objectives'][2]:.2%}")
            print(f"期望 {s['expectations']['n_held']} 成立 / "
                  f"{s['expectations']['n_refuted']} 被推翻")
            print(f"验收 {s['acceptance']['n_passed']}/{s['acceptance']['n']}")
            print(note or "只读：未写盘。")
    if json_mode:
        _emit_json({"cmd": "show", "in_memory_rerun": reran,
                    "note": note or "只读：未写盘。",
                    "summary": products["summary.json"]})
    return 0


_COMMANDS = {"run": cmd_run, "visualize": cmd_visualize, "check": cmd_check,
             "verify": cmd_verify, "show": cmd_show}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = _anchor_args(parser.parse_args(argv))
    return _COMMANDS[args.cmd](args)
