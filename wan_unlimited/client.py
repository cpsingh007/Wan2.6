"""Backends that turn a prompt (plus an optional first-frame image) into one video clip.

`DashScopeBackend` talks to Alibaba Cloud Model Studio (DashScope), which hosts
the Wan2.6 models. `MockBackend` renders a local placeholder clip with ffmpeg so
the whole pipeline can be exercised without an API key or any spend.
"""

from __future__ import annotations

import base64
import mimetypes
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from .ffmpeg import ffmpeg_exe

DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/api/v1"
SYNTHESIS_PATH = "/services/aigc/video-generation/video-synthesis"


class GenerationError(RuntimeError):
    pass


@dataclass
class ClipRequest:
    prompt: str
    duration: int
    image: Path | None = None
    negative_prompt: str | None = None
    seed: int | None = None


def _data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


class DashScopeBackend:
    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        t2v_model: str = "wan2.6-t2v",
        i2v_model: str = "wan2.6-i2v",
        resolution: str = "720P",
        size: str = "1280*720",
        audio: bool = True,
        prompt_extend: bool = True,
        shot_type: str = "single",
        watermark: bool = False,
        poll_interval: float = 10.0,
        timeout: float = 1800.0,
    ):
        if not api_key:
            raise GenerationError("DASHSCOPE_API_KEY is not set")
        self.base_url = base_url.rstrip("/")
        self.t2v_model = t2v_model
        self.i2v_model = i2v_model
        self.resolution = resolution
        self.size = size
        self.audio = audio
        self.prompt_extend = prompt_extend
        self.shot_type = shot_type
        self.watermark = watermark
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {api_key}"

    def _payload(self, req: ClipRequest) -> dict:
        inp: dict = {"prompt": req.prompt}
        params: dict = {
            "duration": req.duration,
            "prompt_extend": self.prompt_extend,
            "shot_type": self.shot_type,
            "audio": self.audio,
            "watermark": self.watermark,
        }
        if req.negative_prompt:
            inp["negative_prompt"] = req.negative_prompt
        if req.seed is not None:
            params["seed"] = req.seed
        if req.image is not None:
            model = self.i2v_model
            inp["img_url"] = _data_uri(req.image)
            params["resolution"] = self.resolution
        else:
            model = self.t2v_model
            params["size"] = self.size
        return {"model": model, "input": inp, "parameters": params}

    def _submit(self, req: ClipRequest) -> str:
        resp = self.session.post(
            self.base_url + SYNTHESIS_PATH,
            json=self._payload(req),
            headers={"X-DashScope-Async": "enable"},
            timeout=120,
        )
        body = resp.json() if resp.content else {}
        if resp.status_code != 200 or "output" not in body:
            raise GenerationError(f"submit failed ({resp.status_code}): {body}")
        return body["output"]["task_id"]

    def _wait(self, task_id: str) -> str:
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            resp = self.session.get(f"{self.base_url}/tasks/{task_id}", timeout=60)
            out = resp.json().get("output", {})
            status = out.get("task_status")
            if status == "SUCCEEDED":
                return out["video_url"]
            if status in ("FAILED", "CANCELED", "UNKNOWN"):
                raise GenerationError(
                    f"task {task_id} {status}: {out.get('code')} {out.get('message')}"
                )
            time.sleep(self.poll_interval)
        raise GenerationError(f"task {task_id} timed out after {self.timeout}s")

    def generate(self, req: ClipRequest, dest: Path) -> Path:
        video_url = self._wait(self._submit(req))
        with requests.get(video_url, stream=True, timeout=300) as r:
            r.raise_for_status()
            tmp = dest.with_suffix(".part")
            with tmp.open("wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        tmp.replace(dest)
        return dest


class MockBackend:
    """Renders a test pattern clip locally; for dry runs and tests."""

    def __init__(self, width: int = 320, height: int = 180, fps: int = 16):
        self.width, self.height, self.fps = width, height, fps

    def generate(self, req: ClipRequest, dest: Path) -> Path:
        size = f"{self.width}x{self.height}"
        if req.image is not None:
            # Continue from the supplied frame so chaining is visible in the output.
            src = ["-loop", "1", "-i", str(req.image)]
            vf = f"scale={size.replace('x', ':')},hue=h=t*40"
        else:
            src = ["-f", "lavfi", "-i", f"testsrc2=size={size}:rate={self.fps}"]
            vf = "null"
        subprocess.run(
            [ffmpeg_exe(), "-y", "-loglevel", "error", *src,
             "-t", str(req.duration), "-vf", vf, "-r", str(self.fps),
             "-pix_fmt", "yuv420p", "-c:v", "libx264", str(dest)],
            check=True,
        )
        return dest
