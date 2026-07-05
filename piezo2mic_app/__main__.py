from __future__ import annotations

import argparse

from .backend_adapter import run_backend_job


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="piezo2mic_app")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--style", choices=("autodetect", "finger", "picked"))
    parser.add_argument("--model", choices=("offline", "realtime"))
    parser.add_argument("--pickup", choices=("piezo", "magnetic", "electric"), default="piezo")

    args = parser.parse_args(argv)

    if args.worker:
        missing = [
            flag
            for flag, value in {
                "--input": args.input,
                "--output": args.output,
                "--style": args.style,
                "--model": args.model,
            }.items()
            if not value
        ]
        if missing:
            parser.error(f"--worker requires: {' '.join(missing)}")

        return run_backend_job(
            input_path=args.input,
            output_path=args.output,
            style=args.style,
            model=args.model,
            pickup=args.pickup,
        )

    try:
        from .ui import launch
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("PySide6"):
            parser.exit(1, "PySide6 is not installed. Run: pip install pyside6\n")
        raise

    return launch()


if __name__ == "__main__":
    raise SystemExit(main())