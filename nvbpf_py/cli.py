from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import types

from .model import ToolSpec
from .render import render_examples_makefile_block, render_tool


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_module(spec_path: Path) -> types.ModuleType:
    module_name = f"nvbpf_py_spec_{spec_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, spec_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load spec from {spec_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _extract_tool_spec(module: types.ModuleType) -> ToolSpec:
    direct = getattr(module, "tool_spec", None)
    if isinstance(direct, ToolSpec):
        return direct

    for value in module.__dict__.values():
        candidate = getattr(value, "tool_spec", None)
        if isinstance(candidate, ToolSpec):
            return candidate

    raise RuntimeError(
        "no ToolSpec found in module. Use @tool(...) on a class with counter(...), "
        "event(...), api_trace(...), @hook/@device_hook methods, or @on_launch_exit."
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate simple NV-BPF tools from Python specs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Generate a tool from a Python spec.")
    build.add_argument("spec", help="Path to the Python spec file.")
    build.add_argument(
        "--output-root",
        default="tools/nvbpf_generated",
        help="Directory under which the tool folder will be generated.",
    )
    build.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing generated tool directory.",
    )
    build.add_argument(
        "--compile",
        action="store_true",
        help="Run `make clean && make <tool>.so` after generating the sources.",
    )
    build.add_argument(
        "--integrate-examples",
        action="store_true",
        help="Write generated sources into tools/nvbpf_examples and patch its Makefile.",
    )

    run = subparsers.add_parser("run", help="Generate, compile, and run a tool from a Python spec.")
    run.add_argument("spec", help="Path to the Python spec file.")
    run.add_argument(
        "--output-root",
        default="tools/nvbpf_generated",
        help="Directory under which the tool folder will be generated.",
    )
    run.add_argument(
        "--no-build",
        action="store_true",
        help="Reuse an existing generated .so instead of rebuilding it first.",
    )
    run.add_argument(
        "cmd",
        nargs=argparse.REMAINDER,
        help="Command to run after '--', for example: -- python3 app.py",
    )
    return parser.parse_args(argv)


def _compile_tool(out_dir: Path, tool_name: str) -> None:
    missing = [name for name in ("nvcc", "ptxas") if shutil.which(name) is None]
    if missing:
        raise RuntimeError(
            "cannot compile generated tool because required CUDA build tools are missing from PATH: "
            + ", ".join(missing)
        )
    subprocess.run(["make", "clean"], cwd=out_dir, check=True)
    subprocess.run(["make", f"{tool_name}.so"], cwd=out_dir, check=True)


def _format_all_block(existing_tools: list[str]) -> str:
    width = 4
    chunks = [existing_tools[i : i + width] for i in range(0, len(existing_tools), width)]
    if len(chunks) == 1:
        return "all: " + " ".join(chunks[0])
    lines = []
    for idx, chunk in enumerate(chunks):
        prefix = "all: " if idx == 0 else "     "
        suffix = " \\" if idx != len(chunks) - 1 else ""
        lines.append(prefix + " ".join(chunk) + suffix)
    return "\n".join(lines)


def _update_examples_makefile(repo_root: Path, tool: ToolSpec) -> None:
    makefile_path = repo_root / "tools/nvbpf_examples/Makefile"
    text = makefile_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    start_idx = None
    end_idx = None
    for i, line in enumerate(lines):
        if line.startswith("all:"):
            start_idx = i
            end_idx = i
            while end_idx + 1 < len(lines) and lines[end_idx].rstrip().endswith("\\"):
                end_idx += 1
            break
    if start_idx is None or end_idx is None:
        raise RuntimeError(f"could not find all: target in {makefile_path}")

    all_tokens: list[str] = []
    for i in range(start_idx, end_idx + 1):
        line = lines[i].replace("\\", "")
        if i == start_idx:
            line = line.split(":", 1)[1]
        all_tokens.extend(tok for tok in line.strip().split() if tok)
    tool_so = f"{tool.name}.so"
    if tool_so not in all_tokens:
        all_tokens.append(tool_so)
    all_block = _format_all_block(all_tokens)
    lines[start_idx : end_idx + 1] = all_block.splitlines()

    begin_marker = f"# -- BEGIN NVBPF_PY {tool.name} --"
    end_marker = f"# -- END NVBPF_PY {tool.name} --"
    block = begin_marker + "\n" + render_examples_makefile_block(tool).rstrip() + "\n" + end_marker
    text = "\n".join(lines)
    if begin_marker in text and end_marker in text:
        block_start = text.index(begin_marker)
        block_end = text.index(end_marker) + len(end_marker)
        text = text[:block_start].rstrip() + "\n\n" + block + text[block_end:]
    else:
        clean_idx = text.find("\nclean:")
        if clean_idx == -1:
            raise RuntimeError(f"could not find clean target in {makefile_path}")
        text = text[:clean_idx].rstrip() + "\n\n" + block + "\n" + text[clean_idx:]

    makefile_path.write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")


def build(
    spec_path: Path,
    output_root: Path,
    force: bool,
    compile_tool: bool,
    integrate_examples: bool,
) -> int:
    repo_root = _repo_root()
    module = _load_module(spec_path)
    tool = _extract_tool_spec(module)
    if integrate_examples:
        out_dir = (repo_root / "tools/nvbpf_examples").resolve()
        rendered = render_tool(tool, out_dir, repo_root, include_makefile=False)
        existing = [out_dir / filename for filename in rendered]
        if any(path.exists() for path in existing) and not force:
            raise RuntimeError(
                f"generated files already exist under {out_dir}\n"
                "use --force to overwrite them"
            )
        for filename, content in rendered.items():
            (out_dir / filename).write_text(content, encoding="utf-8")
        _update_examples_makefile(repo_root, tool)
    else:
        out_dir = (repo_root / output_root / tool.name).resolve()
        if out_dir.exists():
            if not force:
                raise RuntimeError(
                    f"output directory already exists: {out_dir}\n"
                    "use --force to overwrite it"
                )
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        rendered = render_tool(tool, out_dir, repo_root)
        for filename, content in rendered.items():
            (out_dir / filename).write_text(content, encoding="utf-8")

    print(f"generated {tool.name} in {out_dir}")
    print("files:")
    for filename in sorted(rendered):
        print(f"  {filename}")
    print("next:")
    print(f"  cd {out_dir}")
    print(f"  make clean && make {tool.name}.so")

    if compile_tool:
        print("compiling:")
        print(f"  make clean && make {tool.name}.so")
        _compile_tool(out_dir, tool.name)
        print(f"built {out_dir / (tool.name + '.so')}")
    return 0


def run(
    spec_path: Path,
    output_root: Path,
    no_build: bool,
    command: list[str],
) -> int:
    repo_root = _repo_root()
    module = _load_module(spec_path)
    tool = _extract_tool_spec(module)
    out_dir = (repo_root / output_root / tool.name).resolve()
    so_path = out_dir / f"{tool.name}.so"

    if not no_build:
        build(spec_path, output_root, True, True, False)
    elif not so_path.exists():
        raise RuntimeError(
            f"shared object does not exist: {so_path}\n"
            "rerun without --no-build or build the tool first"
        )

    cmd = list(command)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        raise RuntimeError("run requires a command after '--'")

    env = os.environ.copy()
    env.setdefault("ACK_CTX_INIT_LIMITATION", "1")
    existing_preload = env.get("LD_PRELOAD", "")
    env["LD_PRELOAD"] = str(so_path) if not existing_preload else f"{so_path}:{existing_preload}"

    print("running:")
    print(f"  cwd={repo_root}")
    print(f"  LD_PRELOAD={so_path}")
    print(f"  {shlex.join(cmd)}")
    subprocess.run(cmd, cwd=repo_root, env=env, check=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.command == "build":
        return build(
            Path(args.spec),
            Path(args.output_root),
            args.force,
            args.compile,
            args.integrate_examples,
        )
    if args.command == "run":
        return run(
            Path(args.spec),
            Path(args.output_root),
            args.no_build,
            list(args.cmd),
        )
    raise RuntimeError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
