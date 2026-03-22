from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

# Point this at the module that currently owns your CLI.
# Examples:
#   "piezo2mic"
#   "main"
#   "src.piezo2mic"
#
# If your current entrypoint is only a script file, you can also use:
#   "path/to/that_script.py"
#
# Using an importable module name is better if you plan to bundle with PyInstaller.
BACKEND_TARGET = os.environ.get("PIEZO2MIC_BACKEND", "./piezo2mic.py")


def build_backend_argv(
    input_path: str,
    output_path: str,
    style: str,
    model: str,
) -> list[str]:
    return [
        "--process",
        "--input",
        input_path,
        "--output",
        output_path,
        "--style",
        style,
        f"--{model}-only"
    ]


def run_backend_job(
    input_path: str,
    output_path: str,
    style: str,
    model: str,
) -> int:
    argv = build_backend_argv(input_path, output_path, style, model)
    return _run_backend(argv)


def _run_backend(argv: list[str]) -> int:
    old_argv = sys.argv[:]
    old_cwd = os.getcwd()

    try:
        sys.argv = [BACKEND_TARGET, *argv]

        is_path = (
            BACKEND_TARGET.endswith(".py")
            or os.path.sep in BACKEND_TARGET
            or (os.path.altsep and os.path.altsep in BACKEND_TARGET)
        )

        if is_path:
            script_path = Path(BACKEND_TARGET).expanduser().resolve()
            os.chdir(str(script_path.parent))
            runpy.run_path(str(script_path), run_name="__main__")
        else:
            runpy.run_module(BACKEND_TARGET, run_name="__main__")

    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        print(code, file=sys.stderr)
        return 1
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)

    return 0