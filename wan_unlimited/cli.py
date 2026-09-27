"""Command line entry point: `python -m wan_unlimited ...`."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from .client import DEFAULT_BASE_URL, DashScopeBackend, GenerationError, MockBackend
from .local import PRESETS, LocalWanBackend
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
        description="Generate Wan videos of any length by chaining clips.",
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
    p.add_argument("--segment", type=int, default=None,
                   help="Seconds per clip, 2-15 (default 5 local, 10 dashscope).")
    p.add_argument("-o", "--out", type=Path, default=Path("outputs/run"),
                   help="Run directory; reuse it to resume (default outputs/run).")
    p.add_argument("-b", "--backend", choices=["local", "dashscope", "mock"],
                   default=os.environ.get("WAN_BACKEND", "local"),
                   help="local = free open-weights Wan2.2 on your GPU (default); "
                        "dashscope = paid Wan2.6 API; mock = placeholder clips.")
    local = p.add_argument_group("local backend (Wan2.2)")
    local.add_argument("--preset", choices=list(PRESETS),
                       default=os.environ.get("WAN_LOCAL_PRESET", "ti2v-5B"),
                       help="ti2v-5B needs ~24 GB VRAM; a14b needs ~80 GB (default ti2v-5B).")
    local.add_argument("--local-size", help="Override Wan2.2 --size, e.g. 704*1280 for portrait.")
    local.add_argument("--steps", type=int, help="Sampling steps (fewer = faster, lower quality).")
    local.add_argument("--no-low-vram", action="store_true",
                       help="Keep models on GPU (faster, needs more VRAM).")
    api = p.add_argument_group("dashscope backend (Wan2.6 API, paid)")
    p.add_argument("--negative-prompt")
    p.add_argument("--seed", type=int)
    api.add_argument("--resolution", default="720P", choices=["720P", "1080P"])
    api.add_argument("--size", default="1280*720",
                   help="Text-to-video size for the first segment (default 1280*720).")
    api.add_argument("--no-audio", action="store_true", help="Disable generated audio.")
    api.add_argument("--no-prompt-extend", action="store_true")
    api.add_argument("--shot-type", default="single", choices=["single", "multi"])
    api.add_argument("--t2v-model", default=os.environ.get("WAN_T2V_MODEL", "wan2.6-t2v"))
    api.add_argument("--i2v-model", default=os.environ.get("WAN_I2V_MODEL", "wan2.6-i2v"))
    p.add_argument("--stitch-every", type=int, default=None,
                   help="Re-stitch the output every N segments (default 1 with --forever).")
    p.add_argument("--mock", action="store_const", const="mock", dest="backend",
                   help="Shorthand for --backend mock.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _make_backend(args: argparse.Namespace):
    if args.backend == "mock":
        return MockBackend()
    if args.backend == "local":
        return LocalWanBackend(preset=args.preset, size=args.local_size,
                               sample_steps=args.steps, low_vram=not args.no_low_vram)
    return DashScopeBackend(
        api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
        base_url=os.environ.get("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL),
        t2v_model=args.t2v_model, i2v_model=args.i2v_model,
        resolution=args.resolution, size=args.size, audio=not args.no_audio,
        prompt_extend=not args.no_prompt_extend, shot_type=args.shot_type,
    )


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

    if args.segment is None:
        args.segment = 10 if args.backend == "dashscope" else 5
    plan = Plan(prompts=prompts, segment_seconds=args.segment,
                total_seconds=None if args.forever else args.duration,
                start_image=args.image, negative_prompt=args.negative_prompt,
                seed=args.seed)

    try:
        backend = _make_backend(args)
        stitch_every = args.stitch_every if args.stitch_every is not None else int(args.forever)
        final = run(plan, backend, args.out, stitch_every=stitch_every)
    except GenerationError as e:
        logging.error("%s", e)
        return 1
    print(final)
    return 0
