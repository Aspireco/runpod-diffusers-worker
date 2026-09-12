# vid-wan-a14b — Wan 2.2 I2V-A14B (the full-precision 14B MoE)

The tier above the `Wan2.2-TI2V-5B` currently in the stack. Same family, same Apache-2.0
licence, roughly 6x the active parameters and two experts instead of one.

| | deployed 5B | this lane |
| --- | --- | --- |
| repo | `Wan-AI/Wan2.2-TI2V-5B` | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` |
| size on disk | 34.2 GB | **126.2 GB** |
| architecture | single transformer, 5B | **MoE, 2 x 14B-active experts** |
| licence | apache-2.0 | apache-2.0 |
| weights delivered by | Model Caching | **network volume** `yweuz29h2k` (EUR-IS-1) |

## The 126 GB is a storage artefact, not a quality setting

Both experts ship as **fp32** checkpoints — 57.1 GB for 14B parameters is 4 bytes per
parameter. The model was trained and is served in bf16. Loading at `torch_dtype=bfloat16`
is the normal path and halves resident VRAM to ~28.6 GB per expert. That cast is not
quantization and costs nothing measurable; the real quantized arm is `DTYPE=fp8`.

## Deploy

Weights must be on the volume first. This takes roughly an hour and costs about $0.06/hr:

```bash
bash load_weights.sh              # allocates a pod, pulls 126.2 GB, self-terminates
runpodctl pod logs $(cat .loader_pod) | tail
# done when the log prints: WAN_A14B_WEIGHTS_READY_7f3c91
```

Then create the lane. **The network volume pins this endpoint to EUR-IS-1**, which is thin
on stock — it listed only A100 SXM 80GB and RTX PRO 4500 when this was written, both "Low".

```bash
source ../../runpod-stack/.env
TPL=$(runpodctl template create --name mos-wan-a14b --serverless \
  --image ghcr.io/aspireco/runpod-wan-a14b-worker:latest \
  --container-disk-in-gb 30 \
  --env '{"MODEL_DIR":"/runpod-volume/wan22-i2v-a14b","OFFLOAD":"model","DTYPE":"bf16"}' \
  | python -c 'import json,sys; print(json.load(sys.stdin)["id"])')

runpodctl serverless create --name mos-wan-a14b --template-id "$TPL" \
  --gpu-id "NVIDIA A100-SXM4-80GB" \
  --network-volume-id yweuz29h2k \
  --workers-min 0 --workers-max 1 --idle-timeout 60 --execution-timeout 1800
```

Park it at `--workers-max 0` when not benchmarking. Idle workers bill.

## Card sizing

Resident footprint is ~28.6 GB per expert in bf16, plus an 11 GB text encoder.

| card | VRAM | setting | note |
| --- | --- | --- | --- |
| A100 SXM | 80 GB | `OFFLOAD=none` | both experts resident, fastest |
| L40S / A6000 | 48 GB | `OFFLOAD=model` | one expert resident, swapped at the MoE boundary |
| RTX PRO 4500 | 32 GB | `OFFLOAD=model` | tight; drop to 832x480 if it OOMs |
| 4090 | 24 GB | `OFFLOAD=sequential` | works, but layer-by-layer streaming is slow |

## API

```json
{ "input": {
    "prompt": "slow cinematic dolly push-in across a sunlit white oak floor",
    "image_url": "https://.../product-still.jpg",
    "width": 1280, "height": 720,
    "num_frames": 81,
    "num_inference_steps": 40,
    "guidance_scale": 3.5,
    "guidance_scale_2": 3.5,
    "flow_shift": 5.0,
    "seed": 1234
} }
```

This lane is **image-to-video only** — it errors without an image. That is the right shape
for product marketing: the first frame is the real product, so the model animates it rather
than inventing it.

Notes on the inputs that actually matter:

- `num_frames` is forced to the nearest `4n+1`. The temporal VAE encodes in groups of four
  plus a keyframe, so 81 frames is 5.06 s at 16 fps and 80 would be silently truncated.
- `width`/`height` are floored to a multiple of 16 (VAE spatial compression x patch size).
  Values that are not get silently rounded by the pipeline, and the clip then does not match
  the aspect ratio that was requested.
- `guidance_scale_2` is the second expert's guidance. It is only passed if the installed
  diffusers exposes it — an unknown kwarg to a diffusers pipeline is a hard `TypeError`.
- `flow_shift` defaults to 5.0 at 720p and 3.0 below, matching Wan's own reference
  settings. It is the single most effective knob for "moves too much" vs "barely moves".

## Benchmarking

`bench.py` fires one fixed prompt, seed and frame count at any lane so results are
comparable, saves the mp4, and computes cost from measured wall-clock against the card's
posted rate:

```bash
source ../../runpod-stack/.env
python bench.py --lane wan-a14b --endpoint <id> --gpu A100 --image ref/floor.jpg
python bench.py --lane cosmos3  --endpoint pfxkrag2n4j22u --gpu A6000 --image ref/floor.jpg
python bench.py --replay        # markdown table for stack-docs/VIDEO-BENCHMARK.md
```

## Files

| file | what it is |
| --- | --- |
| `handler.py` | the serverless worker |
| `Dockerfile` | image; built by CI as `ghcr.io/aspireco/runpod-wan-a14b-worker` |
| `load_weights.sh` | stages 126.2 GB onto the volume, self-terminating |
| `manifest.tsv` | the 41 files and their exact byte sizes — the download's ground truth |
| `bench.py` | same-prompt harness across lanes |
