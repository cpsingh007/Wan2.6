import json
from pathlib import Path

import pytest

from wan_unlimited import ClipRequest, DashScopeBackend, GenerationError, MockBackend, Plan, run


@pytest.mark.parametrize("total,seg,expected", [
    (60, 10, [10] * 6),
    (21, 10, [10, 9, 2]),
    (23, 5, [5, 5, 5, 5, 3]),
    (15, 15, [15]),
])
def test_durations_hit_total_and_respect_api_minimum(total, seg, expected):
    got = list(Plan(["x"], segment_seconds=seg, total_seconds=total).durations())
    assert got == expected
    assert sum(got) == total and min(got) >= 2


def test_last_prompt_carries_on():
    plan = Plan(["a", "b"])
    assert [plan.prompt_for(i) for i in range(4)] == ["a", "b", "b", "b"]


def test_rejects_segment_outside_api_range():
    with pytest.raises(ValueError):
        Plan(["x"], segment_seconds=20)


class Recorder(MockBackend):
    def __init__(self, fail_at=None):
        super().__init__(width=64, height=64)
        self.calls: list[ClipRequest] = []
        self.fail_at = fail_at

    def generate(self, req, dest):
        if len(self.calls) == self.fail_at:
            raise KeyboardInterrupt
        self.calls.append(req)
        return super().generate(req, dest)


def test_chains_last_frame_and_resumes(tmp_path: Path):
    plan = Plan(["a", "b", "c"], segment_seconds=2, total_seconds=6)

    first = Recorder(fail_at=2)
    run(plan, first, tmp_path)  # interrupted after two segments, partial output stitched
    assert (tmp_path / "final.mp4").exists()
    assert first.calls[0].image is None
    assert first.calls[1].image == tmp_path / "segment_0000_last.png"

    second = Recorder()
    run(plan, second, tmp_path)
    assert [c.prompt for c in second.calls] == ["c"]
    assert second.calls[0].image == tmp_path / "segment_0001_last.png"
    state = json.loads((tmp_path / "state.json").read_text())
    assert [s["duration"] for s in state["segments"]] == [2, 2, 2]


def test_dashscope_payload(tmp_path: Path):
    img = tmp_path / "f.png"
    img.write_bytes(b"\x89PNG")
    b = DashScopeBackend(api_key="sk-test", audio=False)
    t2v = b._payload(ClipRequest(prompt="p", duration=10))
    assert t2v["model"] == "wan2.6-t2v" and t2v["parameters"]["size"] == "1280*720"
    i2v = b._payload(ClipRequest(prompt="p", duration=5, image=img, seed=7))
    assert i2v["model"] == "wan2.6-i2v"
    assert i2v["input"]["img_url"].startswith("data:image/png;base64,")
    assert i2v["parameters"] == {"duration": 5, "prompt_extend": True, "shot_type": "single",
                                 "audio": False, "watermark": False, "seed": 7,
                                 "resolution": "720P"}


def test_dashscope_requires_key():
    with pytest.raises(GenerationError):
        DashScopeBackend(api_key="")
