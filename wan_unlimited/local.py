"""Free, local backend: runs the open-weights Wan2.2 models on your own GPU.

Wan2.5/2.6 are API-only; Wan2.2 is the newest official open-weights release
(Apache 2.0). This backend shells out to Wan2.2's own `generate.py`, which
`scripts/setup_local.sh` clones into `third_party/Wan2.2`.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .client import ClipRequest, GenerationError

log = logging.getLogger("wan_unlimited")

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Preset:
    t2v_task: str
    i2v_task: str
    t2v_ckpt: str
    i2v_ckpt: str
    size: str
    fps: int
    trained_frames: int  # clips longer than this drift in quality


PRESETS = {
    # One 5B model does both text- and image-to-video; fits a 24 GB GPU (e.g. RTX 4090).
    "ti2v-5B": Preset("ti2v-5B", "ti2v-5B", "Wan2.2-TI2V-5B", "Wan2.2-TI2V-5B",
                      "1280*704", 24, 121),
    # Higher quality 14B MoE pair; needs ~80 GB VRAM on a single GPU.
    "a14b": Preset("t2v-A14B", "i2v-A14B", "Wan2.2-T2V-A14B", "Wan2.2-I2V-A14B",
                   "1280*720", 16, 81),
}


def frames_for(seconds: int, fps: int) -> int:
    """Nearest frame count of the form 4n+1, which Wan requires."""
    return max(5, round(seconds * fps / 4) * 4 + 1)


class LocalWanBackend:
    def __init__(
        self,
        preset: str = "ti2v-5B",
        wan_dir: Path | None = None,
        models_dir: Path | None = None,
        size: str | None = None,
        sample_steps: int | None = None,
        low_vram: bool = True,
        python: str = sys.executable,
        extra_args: list[str] | None = None,
    ):
        if preset not in PRESETS:
            raise GenerationError(f"unknown preset {preset!r}; choose from {list(PRESETS)}")
        self.preset = PRESETS[preset]
        self.wan_dir = Path(wan_dir or os.environ.get("WAN_REPO_DIR", ROOT / "third_party/Wan2.2"))
        self.models_dir = Path(models_dir or os.environ.get("WAN_MODELS_DIR", ROOT / "models"))
        self.size = size or self.preset.size
        self.sample_steps = sample_steps
        self.low_vram = low_vram
        self.python = python
        self.extra_args = extra_args or []
        self._warned_length = False

        script = self.wan_dir / "generate.py"
        if not script.exists():
            raise GenerationError(
                f"{script} not found. Run scripts/setup_local.sh first "
                "(or set WAN_REPO_DIR to your Wan2.2 checkout).")
        for ckpt in {self.preset.t2v_ckpt, self.preset.i2v_ckpt}:
            if not (self.models_dir / ckpt).is_dir():
                raise GenerationError(
                    f"model weights not found at {self.models_dir / ckpt}. "
                    "Run scripts/setup_local.sh (or set WAN_MODELS_DIR).")

    def command(self, req: ClipRequest, dest: Path) -> list[str]:
        p = self.preset
        i2v = req.image is not None
        frames = frames_for(req.duration, p.fps)
        if frames > p.trained_frames and not self._warned_length:
            log.warning("%d frames per segment is longer than the %d the model was trained on; "
                        "expect lower quality. --segment %d is recommended.",
                        frames, p.trained_frames, (p.trained_frames - 1) // p.fps)
            self._warned_length = True
        if req.negative_prompt:
            # generate.py has no negative-prompt flag; the model config's default one is used.
            log.debug("negative prompt is not supported by the local backend; ignoring")
        cmd = [
            self.python, "generate.py",
            "--task", p.i2v_task if i2v else p.t2v_task,
            "--ckpt_dir", str((self.models_dir / (p.i2v_ckpt if i2v else p.t2v_ckpt)).resolve()),
            "--size", self.size,
            "--frame_num", str(frames),
            "--prompt", req.prompt,
            "--save_file", str(dest.resolve()),
        ]
        if i2v:
            cmd += ["--image", str(req.image.resolve())]
        if req.seed is not None:
            cmd += ["--base_seed", str(req.seed)]
        if self.sample_steps:
            cmd += ["--sample_steps", str(self.sample_steps)]
        if self.low_vram:
            cmd += ["--offload_model", "True", "--convert_model_dtype", "--t5_cpu"]
        return cmd + self.extra_args

    def generate(self, req: ClipRequest, dest: Path) -> Path:
        cmd = self.command(req, dest)
        log.debug("running %s", " ".join(cmd))
        result = subprocess.run(cmd, cwd=self.wan_dir)
        if result.returncode != 0 or not dest.exists():
            raise GenerationError(f"Wan2.2 generate.py exited with code {result.returncode}")
        return dest
