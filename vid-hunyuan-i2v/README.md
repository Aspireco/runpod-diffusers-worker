# vid-hunyuan-i2v — HunyuanVideo-I2V

A second top-tier image-to-video opinion, and the **hedge** for the A14B lane.

## Why this exists

The brief assumed HunyuanVideo needed a 371.8 GB selective download. It does not — that
figure belongs to `tencent/HunyuanVideo-1.5`, a different and newer model. Measured sizes:

| repo | size | note |
| --- | --- | --- |
| `tencent/HunyuanVideo-I2V` | **30.2 GB** | official, native format |
| `hunyuanvideo-community/HunyuanVideo-I2V` | **43.6 GB** | diffusers port, what this lane loads |
| `tencent/HunyuanVideo` | 39.8 GB | the original T2V 13B |
| `tencent/HunyuanVideo-1.5` | 371.8 GB | where the brief's number came from |

43.6 GB sits inside the Model Caching band, so this lane needs **no network volume**. That
matters for more than cost: `vid-wan-a14b` is pinned to EUR-IS-1 by its volume, and that DC
had only two GPU types in stock with `Low` availability. This lane runs anywhere.

## Licence — read before production use

The loaded repo is a **community re-upload** declaring `base_model: tencent/HunyuanVideo-I2V`,
and it carries **no licence tag of its own**. That changes nothing: the weights are
Tencent's, so the **Tencent Hunyuan Community License** governs.

- **Territory** is worldwide **excluding the European Union, the United Kingdom and South
  Korea**. The **United States is inside the grant** — which is why this is usable where
  MiniMax H3 is not.
- **§4(c) binds Outputs, not just weights:** generated video may not be displayed outside
  the Territory. EU/UK visitors to a public site are the residual exposure.
- §4(b) forbids using Outputs to train any other AI model.
- The 100M-monthly-active-user threshold is irrelevant at this scale.

Narrower than MiniMax H3's restriction (which excludes the US, the primary market), but it
is not Apache. `vid-wan-a14b` remains the licence-clean default.

## Deploy

No volume, no DC pin — the standard Model Caching pattern:

```bash
source ../../runpod-stack/.env
TPL=$(runpodctl template create --name mos-hunyuan-i2v --serverless \
  --image ghcr.io/aspireco/runpod-hunyuan-i2v-worker:latest \
  --container-disk-in-gb 30 \
  --env '{"MODEL_ID":"hunyuanvideo-community/HunyuanVideo-I2V","OFFLOAD":"model"}' \
  | python -c 'import json,sys; print(json.load(sys.stdin)["id"])')

runpodctl serverless create --name mos-hunyuan-i2v --template-id "$TPL" \
  --gpu-id "NVIDIA RTX A6000" \
  --workers-min 0 --workers-max 1 --idle-timeout 60 --execution-timeout 1800 \
  --model-reference https://huggingface.co/hunyuanvideo-community/HunyuanVideo-I2V:main
```

`MODEL_ID` and `--model-reference` must name the same repo. Park at `--workers-max 0` when
not benchmarking — idle workers bill, and the account cap is 10 max-workers total.

## Implementation notes

- **The transformer loads in bf16 while the rest of the pipeline stays fp16.** This is
  diffusers' documented HunyuanVideo configuration and it is not cosmetic: the 13B
  transformer overflows in fp16 on long clips and returns black frames, while the VAE and
  text encoders are fine at fp16 and cheaper there.
- **VAE tiling is on.** The decode is the memory spike, not the denoise — it reconstructs
  every frame at full resolution at once. Tiling is the difference between running and
  OOMing on a 48 GB card at 720p.
- `guidance_scale` defaults to **6.0**. HunyuanVideo is distribution-guided rather than
  CFG-guided and does not behave like a Wan or SDXL guidance scale — do not copy a value
  across from another lane.
- `num_frames` is forced to the nearest `4n+1` (causal temporal VAE), default 121 = ~5 s
  at 24 fps.

## Benchmark

Uses the shared harness so results are comparable:

```bash
python ../vid-wan-a14b/bench.py --lane hunyuan-i2v --endpoint <id> --gpu A6000
```
