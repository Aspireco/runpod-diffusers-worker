# runpod-diffusers-worker

A minimal Runpod serverless worker for diffusers image models. **No weights are baked in.**
Runpod's Model Caching mounts the model at runtime and the handler resolves the snapshot
path itself, so:

- the image builds in minutes and stays far under Runpod's 80 GB / 30-minute build limits
- the endpoint is not pinned to one datacentre the way a network volume pins it
- cold starts are seconds, not the many minutes a fresh 20-60 GB download costs

Change the model by changing one environment variable and one endpoint setting — the code
is model-agnostic.

## Deploy

The image is built by GitHub Actions and published to
`ghcr.io/aspireco/runpod-diffusers-worker:latest`.

```bash
runpodctl template create \
  --name zimage-turbo --serverless \
  --image ghcr.io/aspireco/runpod-diffusers-worker:latest \
  --container-disk-in-gb 30 \
  --env '{"MODEL_ID":"Tongyi-MAI/Z-Image-Turbo"}'

runpodctl serverless create \
  --name mos-zimage \
  --template-id <id from above> \
  --gpu-id "NVIDIA GeForce RTX 4090" \
  --workers-min 0 --workers-max 1 --idle-timeout 60 \
  --model-reference https://huggingface.co/Tongyi-MAI/Z-Image-Turbo:main
```

`MODEL_ID` and `--model-reference` must name the same repo. The first is what the handler
looks for on disk; the second is what Runpod actually caches.

## API

```json
{ "input": {
    "prompt": "a sunlit oak floor in a bright condo",
    "negative_prompt": "blurry, warped text",
    "width": 1024, "height": 1024,
    "num_inference_steps": 8,
    "guidance_scale": 0.0,
    "seed": 42
} }
```

Returns `output.image_url` as a `data:image/png;base64,...` URI, plus the settings used and
`generation_seconds`.

Sampling defaults are per-model (see `DEFAULTS` in `handler.py`) because distilled "turbo"
models want ~8 steps at guidance 0.0, and running them at a normal model's 25 steps and
CFG 7 produces mush.

## Models this has been set up for

| MODEL_ID | Licence | Steps | Notes |
| --- | --- | --- | --- |
| `Tongyi-MAI/Z-Image-Turbo` | Apache-2.0 | 8 | Default. Fast, photoreal, ungated |
| `Qwen/Qwen-Image-2512` | Apache-2.0 | 28 | Best legible in-image text. Needs a 48 GB card at bf16 |
| `black-forest-labs/FLUX.2-klein-4B` | Apache-2.0 | 4 | The only ungated, commercially free FLUX |

All three are ungated — no HuggingFace token needed at any point.

## Gotchas this image already handles

- **torch is not reinstalled.** Default PyPI serves CUDA 13 wheels needing driver >= 580;
  Runpod hosts run 570/575, and a mismatch kills CUDA init before any log line appears.
- **`transformers<5` and `huggingface-hub<1.0`** — the newer majors break diffusers imports.
- **`runpod>=1.10.1`** — 1.7.11 through 1.10.0 corrupt job tracking.
- **The model loads at import**, not in the handler, so FlashBoot captures it in its snapshot.
- **`linux/amd64` is forced** in CI; an arm64 image allocates and then dies on exec.
- **`HF_HUB_OFFLINE=1` is correct here** because the weights really are local. That flag is
  also the true source of the misleading `outgoing traffic has been disabled` error — it
  means the cache was missing, not that Runpod blocked the network.
