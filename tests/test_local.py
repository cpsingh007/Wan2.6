import sys
from pathlib import Path

import pytest

from wan_unlimited import ClipRequest, GenerationError, LocalWanBackend, Plan, run
from wan_unlimited.local import frames_for

# Stands in for Wan2.2's generate.py: records its argv and renders a placeholder clip.
FAKE_GENERATE = """
import argparse, json, pathlib, sys
sys.path.insert(0, {root!r})
from wan_unlimited.client import ClipRequest, MockBackend
p = argparse.ArgumentParser()
for a in ["--task", "--ckpt_dir", "--size", "--frame_num", "--prompt", "--save_file",
          "--image", "--base_seed", "--sample_steps", "--offload_model"]:
    p.add_argument(a)
p.add_argument("--convert_model_dtype", action="store_true")
p.add_argument("--t5_cpu", action="store_true")
a = p.parse_args()
with open("calls.jsonl", "a") as f:
    f.write(json.dumps(vars(a)) + "\\n")
img = pathlib.Path(a.image) if a.image else None
MockBackend(64, 64).generate(ClipRequest(a.prompt, 2, img), pathlib.Path(a.save_file))
"""


@pytest.fixture
def wan(tmp_path):
    wan_dir, models = tmp_path / "Wan2.2", tmp_path / "models"
    wan_dir.mkdir()
    (models / "Wan2.2-TI2V-5B").mkdir(parents=True)
    root = str(Path(__file__).resolve().parent.parent)
    (wan_dir / "generate.py").write_text(FAKE_GENERATE.format(root=root))
    return wan_dir, models


def test_frames_are_4n_plus_1():
    assert frames_for(5, 24) == 121 and frames_for(5, 16) == 81
    assert all(frames_for(s, 24) % 4 == 1 for s in range(2, 16))


def test_missing_setup_is_explained(tmp_path):
    with pytest.raises(GenerationError, match="setup_local.sh"):
        LocalWanBackend(wan_dir=tmp_path, models_dir=tmp_path)


def test_command_uses_ti2v_task_for_both_modes(wan, tmp_path):
    b = LocalWanBackend(wan_dir=wan[0], models_dir=wan[1])
    t2v = b.command(ClipRequest("p", 5, seed=3), tmp_path / "o.mp4")
    assert t2v[t2v.index("--task") + 1] == "ti2v-5B"
    assert t2v[t2v.index("--frame_num") + 1] == "121"
    assert t2v[t2v.index("--base_seed") + 1] == "3"
    assert "--image" not in t2v and "--t5_cpu" in t2v
    i2v = b.command(ClipRequest("p", 5, image=tmp_path / "f.png"), tmp_path / "o.mp4")
    assert i2v[i2v.index("--image") + 1] == str(tmp_path / "f.png")


def test_end_to_end_chaining(wan, tmp_path):
    import json
    b = LocalWanBackend(wan_dir=wan[0], models_dir=wan[1], python=sys.executable)
    out = tmp_path / "run"
    final = run(Plan(["a", "b"], segment_seconds=5, total_seconds=15), b, out)
    assert final.exists()
    calls = [json.loads(l) for l in (wan[0] / "calls.jsonl").read_text().splitlines()]
    assert [c["image"] for c in calls] == [
        None, str(out.resolve() / "segment_0000_last.png"), str(out.resolve() / "segment_0001_last.png")]
    assert [c["prompt"] for c in calls] == ["a", "b", "b"]
