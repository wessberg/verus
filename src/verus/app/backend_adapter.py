from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

# The backend is the `verus` CLI (python -m verus). Override with VERUS_BACKEND to point at another
# importable module name or a script path.
BACKEND_TARGET = os.environ.get("VERUS_BACKEND", "verus")

# The CLI resolves input/ and out/ relative to the working directory, so jobs run from the project
# root: $VERUS_HOME if set, else the checkout this file lives in, else the current directory.
def _project_root() -> Path:
    env = os.environ.get("VERUS_HOME")
    if env:
        return Path(env).expanduser().resolve()
    checkout = Path(__file__).resolve().parents[3]
    return checkout if (checkout / "pyproject.toml").exists() else Path.cwd()


def build_backend_argv(
    input_path: str,
    output_path: str,
    style: str,
    model: str,
    pickup: str = "piezo",
) -> list[str]:
    return [
        "--process",
        "--input",
        input_path,
        "--output",
        output_path,
        "--style",
        style,
        "--pickup",
        pickup,
        f"--{model}-only"
    ]


def run_backend_job(
    input_path: str,
    output_path: str,
    style: str,
    model: str,
    pickup: str = "piezo",
) -> int:
    argv = build_backend_argv(input_path, output_path, style, model, pickup)
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
            os.chdir(str(_project_root()))
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