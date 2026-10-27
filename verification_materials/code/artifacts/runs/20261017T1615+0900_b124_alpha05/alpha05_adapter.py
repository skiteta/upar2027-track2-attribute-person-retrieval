"""Thin fixed B124 CLI. Root alone admits any build, replay or reference."""
import argparse
from pathlib import Path

from upar.fixed_alpha_revision import FixedAlphaRevision

HERE = Path(__file__).resolve().parent


def recipe():
    return FixedAlphaRevision(HERE / "recipe_v1.json", Path(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("build", "replay", "reference"))
    parser.add_argument("--root-reviewed", action="store_true", required=True)
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--package-report", type=Path)
    parser.add_argument("--build-exit", type=Path)
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--replay-exit", type=Path)
    parser.add_argument("--time-scale", type=float, default=1.)
    args = parser.parse_args()
    r = recipe()
    if args.mode == "build":
        if args.allow_gpu:
            parser.error("CPU-only build")
        r.build(args.output)
    elif args.mode == "replay":
        if not args.allow_gpu or not args.package_report or not args.build_exit:
            parser.error("root GPU admission plus actual new build exit required")
        r.replay(args.package_report, args.build_exit, args.output, args.time_scale)
    else:
        if args.allow_gpu or not args.capture or not args.replay_exit:
            parser.error("CPU-only reference requires new capture actual exit")
        r.reference(args.capture, args.replay_exit, args.output)


if __name__ == "__main__":
    main()
