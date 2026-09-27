# Wan2.6

Generate videos of any length with Alibaba's **Wan2.6** models.

A single Wan2.6 request returns at most **15 seconds** of video. `wan_unlimited`
removes that ceiling by chaining requests:

1. Segment 1 is generated with text-to-video (`wan2.6-t2v`), or image-to-video if you pass `--image`.
2. The last frame of each segment becomes the first frame of the next (`wan2.6-i2v`), so motion and subjects carry over.
3. The segments are joined into one `final.mp4` with ffmpeg.

You can give a single prompt or a storyboard with one prompt per segment. Progress
is saved after every segment, so an interrupted run resumes where it stopped.

> "Unlimited" means unlimited **length**. Every segment is a normal, billed API call
> on your Alibaba Cloud Model Studio account, and your account's quotas and rate
> limits still apply. A 60 s video at the default 10 s per segment is 6 calls.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # includes a bundled ffmpeg binary
cp .env.example .env                     # then put your DASHSCOPE_API_KEY in .env
```

Get an API key from [Alibaba Cloud Model Studio](https://modelstudio.console.alibabacloud.com/).
The default endpoint is the international (Singapore) region. For Beijing, set
`DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/api/v1`.

Try the whole pipeline with no key and no cost (it renders placeholder clips locally):

```bash
python -m wan_unlimited --mock -s examples/storyboard.txt -d 40
```

## Usage

```bash
# 2-minute video from one prompt
python -m wan_unlimited -p "A paper boat drifting down a rainy city gutter, macro shot" -d 120

# Storyboard: one prompt per segment; the last prompt keeps going once they run out
python -m wan_unlimited -s examples/storyboard.txt -d 90 -o outputs/hike

# Start from your own image, in 1080P, with 15 s segments
python -m wan_unlimited -i start.png -p "The scene slowly comes alive" -d 300 --segment 15 --resolution 1080P

# Keep generating until you press Ctrl-C; final.mp4 is re-stitched after every segment
python -m wan_unlimited -p "An endless drive through neon-lit Tokyo at night" --forever -o outputs/drive
```

To resume after a crash, Ctrl-C or a network failure, run the same command with the
same `-o` directory. Segments already listed in `state.json` are skipped.

### Options

| Flag | Default | Meaning |
|---|---|---|
| `-p/--prompt` | – | Prompt; repeat the flag to build a storyboard |
| `-s/--storyboard` | – | `.txt` (one prompt per line, `#` for comments) or `.json` list |
| `-i/--image` | – | First-frame image for segment 1 |
| `-d/--duration` | `60` | Total seconds |
| `--forever` | off | Run until stopped instead of to a fixed duration |
| `--segment` | `10` | Seconds per API call (2–15) |
| `-o/--out` | `outputs/run` | Run directory (reuse it to resume) |
| `--resolution` | `720P` | `720P` or `1080P` for image-to-video segments |
| `--size` | `1280*720` | Size of the first text-to-video segment |
| `--no-audio` | off | Turn off Wan2.6's generated audio (lowers cost) |
| `--shot-type` | `single` | `multi` lets Wan cut between shots within a segment |
| `--seed`, `--negative-prompt` | – | Passed through to every segment |
| `--t2v-model`, `--i2v-model` | `wan2.6-t2v`, `wan2.6-i2v` | e.g. `wan2.6-i2v-flash` for cheaper continuations |
| `--stitch-every N` | `0` (`1` with `--forever`) | Re-stitch `final.mp4` every N segments |
| `--mock` | off | Local placeholder generator, for testing |

## Tips for long videos

- Keep `--shot-type single` for smooth continuity. Hard cuts inside a segment make the next segment start from an unrelated frame.
- Describe *what happens next* in each storyboard line rather than repeating the whole scene. The image already carries the look.
- A fixed `--seed` and a `--negative-prompt` (e.g. `"blurry, distorted, text, watermark"`) help reduce drift over many segments.
- Image-to-video output sizes can differ slightly from the first text-to-video clip. If so, the stitcher falls back from stream copy to re-encoding automatically. Start from `--image` to keep every segment identical.

## Python API

```python
from pathlib import Path
from wan_unlimited import DashScopeBackend, Plan, run

plan = Plan(prompts=["A fox runs through snow", "The fox leaps into a snowbank"],
            segment_seconds=10, total_seconds=120)
run(plan, DashScopeBackend(api_key="sk-..."), Path("outputs/fox"))
```

## Tests

```bash
pip install pytest && python -m pytest -q
```
