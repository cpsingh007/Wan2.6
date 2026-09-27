"""Thin ffmpeg helpers: grab a clip's last frame and join clips into one video."""

from __future__ import annotations

import shutil
import subprocess
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=None)
def ffmpeg_exe() -> str:
    system = shutil.which("ffmpeg")
    if system:
        return system
    import imageio_ffmpeg  # bundled static binary, installed via requirements.txt

    return imageio_ffmpeg.get_ffmpeg_exe()


def _run(args: list[str]) -> None:
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", *args], check=True)


def last_frame(video: Path, dest: Path) -> Path:
    """Write the final frame of `video` to `dest` (PNG)."""
    # Seek near the end and keep overwriting the output so the last decoded frame wins.
    _run(["-sseof", "-1", "-i", str(video), "-update", "1", "-q:v", "1", str(dest)])
    if not dest.exists():
        _run(["-i", str(video), "-update", "1", "-q:v", "1", str(dest)])
    return dest


def concat(clips: list[Path], dest: Path) -> Path:
    """Join clips in order. Tries a lossless stream copy, then falls back to re-encoding."""
    listing = dest.with_suffix(".txt")
    listing.write_text("".join(f"file '{c.resolve()}'\n" for c in clips))
    try:
        _run(["-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(dest)])
    except subprocess.CalledProcessError:
        _run(["-f", "concat", "-safe", "0", "-i", str(listing),
              "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
              "-c:a", "aac", "-b:a", "192k", str(dest)])
    finally:
        listing.unlink(missing_ok=True)
    return dest
