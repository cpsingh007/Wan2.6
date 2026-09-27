# Wan2.6

Generate videos of any length with Wan, for free on your own GPU.

**Which model does it use?** Wan2.5 and Wan2.6 are **not open source**. Alibaba
only offers them through its paid API. The newest official open-weights release is
**Wan2.2** (Apache 2.0), so that's what the free setup runs. The paid Wan2.6 API is
still supported as an optional backend.

| Backend | Model | Cost | Needs |
|---|---|---|---|
| `local` (default) | Wan2.2 TI2V-5B or A14B | **Free**, unlimited | NVIDIA GPU (24 GB for 5B, 80 GB for A14B) |
| `dashscope` | Wan2.6 | Paid per second of video | API key |
| `mock` | placeholder clips | Free | nothing (for testing) |

## How it makes long videos

Each model call produces a short clip, about 5 seconds for Wan2.2. `wan_unlimited`
chains them:

1. Clip 1 comes from your text prompt, or from your image if you pass `--image`.
2. Each later clip starts from the **last frame of the clip before it**, so subjects and motion continue.
3. All clips are joined into `final.mp4`.

Progress is saved after every clip. If a run stops, the same command resumes it.

## Free setup (local Wan2.2)

Requirements: Linux (or WSL2), an NVIDIA GPU with CUDA drivers, Python 3.10+, git,
and about 40 GB of free disk.

```bash
git clone <this repo> && cd Wan2.6
python3 -m venv .venv && source .venv/bin/activate
bash scripts/setup_local.sh          # TI2V-5B: clones Wan2.2, installs deps, downloads ~35 GB of weights
```

The script:
1. Clones [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2) into `third_party/`.
2. Installs PyTorch, Wan2.2's requirements and flash-attn.
3. Downloads [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) into `models/`.

If you have an 80 GB GPU (A100/H100), `bash scripts/setup_local.sh a14b` gets the
higher-quality 14B models instead, then add `--preset a14b` when you generate.

Check that everything works without touching the GPU:

```bash
python -m wan_unlimited --mock -s examples/storyboard.txt -d 20
```

### Generating

```bash
# 1-minute video from one prompt (12 clips of 5 s)
python -m wan_unlimited -p "A red fox trotting through fresh snow at dawn, cinematic" -d 60

# Storyboard: one prompt per clip; the last prompt keeps going once they run out
python -m wan_unlimited -s examples/storyboard.txt -d 120 -o outputs/hike

# Start from your own picture, portrait orientation
python -m wan_unlimited -i start.png -p "She turns and walks into the crowd" -d 45 --local-size 704*1280

# Keep going until Ctrl-C; final.mp4 is updated after every clip
python -m wan_unlimited -p "An endless drive through neon-lit Tokyo at night" --forever -o outputs/drive
```

To resume after a crash or Ctrl-C, run the same command with the same `-o` folder.

**Speed:** with TI2V-5B, one 5-second 720p clip takes about 9 minutes on an RTX 4090,
so a 1-minute video takes about 2 hours. `--steps 30` (default 50) trades some quality
for speed. If you have VRAM to spare, `--no-low-vram` keeps the models on the GPU.

### No suitable GPU?

- **Rent one** by the hour (RunPod, Vast.ai, Lambda). This isn't free, but you only pay for GPU time, not per second of video. Pick a 24 GB+ card, clone this repo and run the setup script.
- **Free notebooks** (Google Colab, Kaggle) give you 16 GB GPUs. That is below Wan2.2's requirements, so expect out-of-memory errors.
- Use `--backend dashscope` below for Wan2.6 quality with no GPU (paid).

## Optional: paid Wan2.6 API

```bash
cp .env.example .env    # set WAN_BACKEND=dashscope and DASHSCOPE_API_KEY
python -m wan_unlimited -b dashscope -p "..." -d 60        # 10 s per call by default, up to 15
```

Every clip is a billed call on your Alibaba Cloud Model Studio account.
Dashscope-only flags: `--resolution 720P|1080P`, `--size`, `--no-audio`,
`--shot-type single|multi`, `--no-prompt-extend`, `--t2v-model`, `--i2v-model`.

## All options

| Flag | Default | Meaning |
|---|---|---|
| `-p/--prompt` | – | Prompt; repeat the flag to build a storyboard |
| `-s/--storyboard` | – | `.txt` (one prompt per line, `#` for comments) or `.json` list |
| `-i/--image` | – | First-frame image for clip 1 |
| `-d/--duration` | `60` | Total seconds |
| `--forever` | off | Run until stopped instead of to a fixed duration |
| `--segment` | `5` local / `10` API | Seconds per clip (2–15). Local models were trained on 5 s clips; longer ones lose quality |
| `-o/--out` | `outputs/run` | Run folder (reuse it to resume) |
| `-b/--backend` | `local` | `local`, `dashscope` or `mock` |
| `--preset` | `ti2v-5B` | Local model: `ti2v-5B` or `a14b` |
| `--local-size` | model default | `1280*704`/`704*1280` (5B); `1280*720`, `720*1280`, `832*480`, `480*832` (A14B) |
| `--steps` | model default | Local sampling steps |
| `--no-low-vram` | off | Keep local models on GPU |
| `--seed` | random | Fixed seed for every clip (helps consistency) |
| `--stitch-every N` | `0` (`1` with `--forever`) | Re-stitch `final.mp4` every N clips |

## Tips for long videos

- In each storyboard line, describe *what happens next* rather than repeating the whole scene. The previous frame already carries the look.
- A fixed `--seed` reduces drift over many clips.
- Quality slowly drifts over very long chains (colors, faces). Splitting a story into scenes, each started from a fresh `--image`, keeps it sharp.

## Python API

```python
from pathlib import Path
from wan_unlimited import LocalWanBackend, Plan, run

plan = Plan(prompts=["A fox runs through snow", "The fox leaps into a snowbank"],
            segment_seconds=5, total_seconds=120)
run(plan, LocalWanBackend(preset="ti2v-5B"), Path("outputs/fox"))
```

## Tests

```bash
pip install pytest && python -m pytest -q
```
