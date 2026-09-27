"""Chain Wan clips into a video of any length.

Each Wan2.6 call returns at most 15 s. To go longer, every segment after the
first is generated image-to-video from the previous segment's last frame, then
all segments are concatenated. Progress is written to `state.json` after every
segment, so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import itertools
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Protocol

from . import ffmpeg
from .client import ClipRequest, GenerationError

log = logging.getLogger("wan_unlimited")

MIN_SEGMENT, MAX_SEGMENT = 2, 15


class Backend(Protocol):
    def generate(self, req: ClipRequest, dest: Path) -> Path: ...


@dataclass
class Plan:
    prompts: list[str]
    segment_seconds: int = 10
    total_seconds: int | None = None  # None = run until stopped
    start_image: Path | None = None
    negative_prompt: str | None = None
    seed: int | None = None
    max_retries: int = 3
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.prompts:
            raise ValueError("at least one prompt is required")
        if not MIN_SEGMENT <= self.segment_seconds <= MAX_SEGMENT:
            raise ValueError(f"segment length must be {MIN_SEGMENT}-{MAX_SEGMENT} s")

    def durations(self) -> Iterator[int]:
        """Per-segment durations; the last one is shortened to hit the total exactly."""
        if self.total_seconds is None:
            yield from itertools.repeat(self.segment_seconds)
            return
        remaining = self.total_seconds
        while remaining > 0:
            d = min(self.segment_seconds, remaining)
            if remaining - d and remaining - d < MIN_SEGMENT:
                d = remaining - MIN_SEGMENT  # avoid a final clip shorter than the API allows
            yield max(d, MIN_SEGMENT)
            remaining -= d

    def prompt_for(self, index: int) -> str:
        """Storyboard prompt for segment `index`; the last prompt carries on once they run out."""
        return self.prompts[min(index, len(self.prompts) - 1)]


def _load_state(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {"segments": []}


def _save_state(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(path)


def run(plan: Plan, backend: Backend, out_dir: Path, output_name: str = "final.mp4",
        stitch_every: int = 0) -> Path:
    """Generate every segment of `plan` into `out_dir` and stitch them into one file.

    `stitch_every` > 0 re-stitches the output after every N new segments, so an
    open-ended run always has an up-to-date video on disk.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "state.json"
    state = _load_state(state_path)
    done: list[dict] = state["segments"]
    final = out_dir / output_name

    if done:
        log.info("resuming: %d segment(s) already generated", len(done))

    try:
        for i, duration in enumerate(plan.durations()):
            if i < len(done):
                continue
            clip = out_dir / f"segment_{i:04d}.mp4"
            if i == 0:
                image = plan.start_image
            else:
                image = ffmpeg.last_frame(out_dir / done[-1]["file"],
                                          out_dir / f"segment_{i - 1:04d}_last.png")
            req = ClipRequest(prompt=plan.prompt_for(i), duration=duration, image=image,
                              negative_prompt=plan.negative_prompt, seed=plan.seed)
            _generate_with_retry(backend, req, clip, plan.max_retries, i)
            done.append({"index": i, "file": clip.name, "duration": duration,
                         "prompt": req.prompt})
            _save_state(state_path, state)
            log.info("segment %d done (%ds, %ds total)", i, duration,
                     sum(s["duration"] for s in done))
            if stitch_every and len(done) % stitch_every == 0:
                ffmpeg.concat([out_dir / s["file"] for s in done], final)
                log.info("stitched %d segments -> %s", len(done), final)
    except KeyboardInterrupt:
        log.warning("interrupted; stitching what exists so far (rerun to resume)")

    if not done:
        raise GenerationError("no segments were generated")
    ffmpeg.concat([out_dir / s["file"] for s in done], final)
    log.info("wrote %s (%d segments, %ds)", final, len(done),
             sum(s["duration"] for s in done))
    return final


def _generate_with_retry(backend: Backend, req: ClipRequest, dest: Path,
                         retries: int, index: int) -> None:
    for attempt in range(1, retries + 1):
        try:
            log.info("segment %d attempt %d: %s", index, attempt, req.prompt[:80])
            backend.generate(req, dest)
            return
        except (GenerationError, OSError) as e:
            if attempt == retries:
                raise
            wait = 2 ** attempt * 5
            log.warning("segment %d failed (%s); retrying in %ds", index, e, wait)
            time.sleep(wait)
