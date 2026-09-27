"""Command line entry point: `python -m wan_unlimited ...`."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from .client import DEFAULT_BASE_URL, DashScopeBackend, MockBackend
from .pipeline import Plan, run


def _load_prompts(args: argparse.Namespace) -> list[str]:
    prompts = list(args.prompt or [])
    if args.storyboard:
        text = Path(args.storyboard).read_text()
        if args.storyboard.endswith(".json"):
            prompts += [str(p) for p in json.loads(text)]
        else:
            prompts += [ln.strip() for ln in text.splitlines()
                        if ln.strip() and not ln.lstrip().startswith("#")]
    return prompts


def _load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep and not key.strip().startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="wan_unlimited",
        description="Generate Wan2.6 videos of any length by chaining clips.",
    )
    p.add_argument("-p", "--prompt", action="append",
                   help="Prompt for the next segment; repeat for a storyboard.")
    p.add_argument("-s", "--storyboard",
                   help="Text file (one prompt per line) or JSON list of prompts.")
    p.add_argument("-i", "--image", type=Path, help="Optional first-frame image.")
    length = p.add_mutually_exclusive_group()
    length.add_argument("-d", "--duration", type=int, default=60,
                        help="Total length in seconds (default 60).")
    length.add_argument("--forever", action="store_true",
                        help="Keep generating until Ctrl-C; the output stays playable.")
    p.add_argument("--segment", type=int, default=10,
                   help="Seconds per API call, 2-15 (default 10).")
    p.add_argument("-o", "--out", type=Path, default=Path("outputs/run"),
                   help="Run directory; reuse it to resume (default outputs/run).")
    p.add_argument("--negative-prompt")
    p.add_argument("--seed", type=int)
    p.add_argument("--resolution", default="720P", choices=["720P", "1080P"])
    p.add_argument("--size", default="1280*720",
                   help="Text-to-video size for the first segment (default 1280*720).")
    p.add_argument("--no-audio", action="store_true", help="Disable generated audio.")
    p.add_argument("--no-prompt-extend", action="store_true")
    p.add_argument("--shot-type", default="single", choices=["single", "multi"])
    p.add_argument("--t2v-model", default=os.environ.get("WAN_T2V_MODEL", "wan2.6-t2v"))
    p.add_argument("--i2v-model", default=os.environ.get("WAN_I2V_MODEL", "wan2.6-i2v"))
    p.add_argument("--stitch-every", type=int, default=None,
                   help="Re-stitch the output every N segments (default 1 with --forever).")
    p.add_argument("--mock", action="store_true",
                   help="Use a local placeholder generator (no API key, no cost).")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    prompts = _load_prompts(args)
    if not prompts:
        build_parser().error("give at least one --prompt or a --storyboard")
    if not args.forever and args.duration < 2:
        build_parser().error("--duration must be at least 2 seconds")

    plan = Plan(prompts=prompts, segment_seconds=args.segment,
                total_seconds=None if args.forever else args.duration,
                start_image=args.image, negative_prompt=args.negative_prompt,
                seed=args.seed)

    if args.mock:
        backend = MockBackend()
    else:
        backend = DashScopeBackend(
            api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
            base_url=os.environ.get("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL),
            t2v_model=args.t2v_model, i2v_model=args.i2v_model,
            resolution=args.resolution, size=args.size, audio=not args.no_audio,
            prompt_extend=not args.no_prompt_extend, shot_type=args.shot_type,
        )

    stitch_every = args.stitch_every if args.stitch_every is not None else int(args.forever)
    final = run(plan, backend, args.out, stitch_every=stitch_every)
    print(final)
    return 0
