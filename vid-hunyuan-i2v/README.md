# vid-hunyuan-i2v — HunyuanVideo-I2V

A second top-tier image-to-video opinion alongside `vid-wan-a14b`, and the hedge for it. Azure
GPU fleet lane `hunyuan-i2v`, A100 80GB PCIe.

## Premade search

| Need | Searched | Using | Why |
|---|---|---|---|
| The model and its generation code | The original `tencent/HunyuanVideo-I2V` native checkpoint vs. the `hunyuanvideo-community/HunyuanVideo-I2V` diffusers port | **The diffusers port**, through `diffusers.HunyuanVideoImageToVideoPipeline` (public API, no vendored model code) | This worker pre-existed as a RunPod worker (see git history) already built on this exact port and pipeline; it is diffusers' own documented recipe for the model, not a reimplementation. Nothing here needed to be built from scratch. |
| Weight staging on Azure | RunPod's Model Caching (the original worker's mechanism: a `MODEL_ID` env var plus a glob-search of a HF-shaped cache under `/runpod-volume/huggingface-cache/hub/`) | **The shared Azure fetch job**, `gpu/manifests/hunyuan-i2v.tsv`, same flat-directory convention as every other Azure lane | RunPod Model Caching does not exist on Azure Container Apps. The flat-directory + `local_files_only=True` pattern is already proven on `vid-wan-animate` and `vid-ltx25`; no new mechanism needed. |

Nothing here was built from scratch. The only work was porting an already-working RunPod worker
to the Azure lane contract: swap the weight source, add `build_check.py`, update the README.

## Licence — read before production use

The loaded repo (`hunyuanvideo-community/HunyuanVideo-I2V`) is a **community re-upload**
declaring `base_model: tencent/HunyuanVideo-I2V`, with **no licence tag of its own** on the
Hugging Face repo card. A missing tag on a re-upload changes nothing: the weights are Tencent's,
so the governing terms are the **Tencent Hunyuan Community License** published with
`tencent/HunyuanVideo-I2V` itself (confirmed by reading that LICENSE file directly, not the
re-upload's card).

| Component | Source | Licence | Territory / output restriction |
|---|---|---|---|
| HunyuanVideo-I2V code path | `diffusers` (Apache-2.0), calling the model's own architecture | Apache-2.0 | n/a — this is generic pipeline code, not Tencent's |
| HunyuanVideo-I2V weights | `hunyuanvideo-community/HunyuanVideo-I2V` (re-upload of `tencent/HunyuanVideo-I2V`) | **Tencent Hunyuan Community License** | Territory = worldwide **excluding the EU, UK and South Korea** (LICENSE §1(l)). The **United States is inside the grant** — this is why this lane is usable where a EU/UK/Korea-exposed lane would not be. |
| Generated video (Outputs) | — | Same licence, §5(c) | **§5(c) binds Outputs, not just weights**: "You must not use, reproduce, modify, distribute, or display the Tencent Hunyuan Works, Output or results of the Tencent Hunyuan Works outside the Territory." EU/UK/South Korea visitors to a public site are the residual exposure — this is the reason this image is built only in `azswmcacr` (East US 2) and runs only in the Canada Central GPU environment. |
| Training other models on Outputs | — | Same licence, §5(b) | "You must not use the Tencent Hunyuan Works or any Output or results of the Tencent Hunyuan Works to improve any other AI model (other than Tencent Hunyuan or Model Derivatives thereof)." Outputs from this lane may not be used as training data for any other model. |
| Scale threshold | — | Same licence, §4 | Above **100,000,000 monthly active users** (all products/services, combined), a separate licence must be requested from Tencent; irrelevant at this scale but recorded per the task brief. |

All four clauses verified by fetching `https://huggingface.co/tencent/HunyuanVideo-I2V/raw/main/LICENSE`
directly on 2026-10-18, not assumed from the brief or the re-upload's card.

## Handler contract

Input (`job["input"]`):

```json
{
  "image": "data:image/png;base64,... or raw base64 or https://...",
  "prompt": "a required text prompt",
  "width": 1280,
  "height": 720,
  "num_frames": 129,
  "num_inference_steps": 30,
  "guidance_scale": 6.0,
  "seed": -1
}
```

- `image` (also accepted as `image_base64` or `image_url`): a data URL, raw base64, or an
  `https://` URL. **URL fetches go through `safe_fetch.py`, an SSRF guard** — job inputs can
  arrive from an LLM agent, not just a human, so a URL must never be able to reach an internal
  address (cloud metadata, localhost, RFC1918, link-local). `http://` is refused outright; an
  `https://` host is resolved and every resolved address must be public, then the connection goes
  to that validated address directly (not re-resolved), with TLS SNI still the original hostname.
  **Redirects are refused, not followed** — a URL from a host that 302s (GitHub release assets do
  this; some signed-URL patterns do too) comes back as `{"error": "could not load image:
  redirects are not followed; fetch the final URL directly"}`. Sending the image inline as a data
  URL always works and has none of this limitation.
- `num_frames` (default **129**, ~5.4s at the model's native 24fps): rounded to the nearest valid
  `4n+1` value — the causal temporal VAE encodes in groups of four plus a keyframe. 129 is itself
  already `4*32+1`. (The pre-Azure RunPod worker defaulted to 121; 129 is the figure in this task's
  own brief and is used here instead.)
- `num_inference_steps` (default 30) and `guidance_scale` (default 6.0): unchanged from the
  pre-Azure worker's own already-researched defaults — HunyuanVideo is distribution-guided rather
  than CFG-guided, so 6.0 is not comparable to a Wan/SDXL guidance scale.
- `seed`: -1 (default) picks a random seed; otherwise deterministic.

Output:

```json
{
  "video_url": "data:video/mp4;base64,...",
  "width": 1280,
  "height": 720,
  "fps": 24,
  "frames": 129,
  "seconds": 5.38,
  "seed": 1234567890,
  "generation_seconds": 210.4,
  "model": "hunyuanvideo-community/HunyuanVideo-I2V",
  "licence": "Tencent Hunyuan Community (Territory excludes EU/UK/South Korea; binds Outputs, not just weights)",
  "dtype": "bf16-transformer/fp16-rest",
  "steps": 30
}
```

On failure: `{"error": "<Type>: <message>"}`, never a raised exception.

## VRAM reasoning on an A100 80GB (reasoned, not benchmarked — no GPU job was started)

| Component | dtype | approx. size |
|---|---|---|
| Transformer (13B) | bf16 (diffusers' documented config for this model — the transformer overflows in fp16 on long clips and returns black frames) | ~26GB |
| Text encoders (2) + VAE | fp16 | ~5GB combined |
| Activations, 129 frames @ 720p | fp16 | a few GB, bounded further by VAE tiling on decode |

Comfortably under 80GB even fully GPU-resident, with more headroom than the pre-Azure worker's
original target (RTX A6000, 48GB). `OFFLOAD=model` (model-level CPU offload) and VAE tiling are
kept on anyway: they were the already-proven settings on the smaller card, cost only a little
extra latency per job (no extra GPU-seconds once the swap overlaps with compute), and relaxing
them has no measured upside without a GPU job to time it — the task scope excludes starting one.
`OFFLOAD=sequential` remains available as an env var for a future smaller GPU type.

## Files under `$HUNYUAN_I2V_DIR` (default `/runpod-volume/hunyuan-i2v`)

Exactly the diffusers repo `hunyuanvideo-community/HunyuanVideo-I2V`'s own layout —
`gpu/manifests/hunyuan-i2v.tsv` (28 files, 43.65 GB) is everything the handler loads (`.gitattributes`
and `README.md` excluded, nothing in the handler's import path touches them).

```
$HUNYUAN_I2V_DIR/
├── model_index.json
├── image_processor/preprocessor_config.json
├── scheduler/scheduler_config.json
├── text_encoder/{config,generation_config}.json, model-0000{1..4}-of-00004.safetensors, model.safetensors.index.json
├── text_encoder_2/{config.json, model.safetensors}
├── tokenizer/{special_tokens_map,tokenizer_config}.json, tokenizer.json
├── tokenizer_2/{merges.txt, special_tokens_map.json, tokenizer_config.json, vocab.json}
├── transformer/config.json, diffusion_pytorch_model-0000{1..5}-of-00005.safetensors, diffusion_pytorch_model.safetensors.index.json
└── vae/{config.json, diffusion_pytorch_model.safetensors}
```

Wire the fetch job with repo id `hunyuanvideo-community/HunyuanVideo-I2V` and destination
`$HUNYUAN_I2V_DIR` against `gpu/manifests/hunyuan-i2v.tsv`.

## What changed from the pre-Azure (RunPod) worker

- **Weight source**: RunPod's Model Caching (`MODEL_ID` env var + `find_snapshot()` glob-searching
  a HF-cache-shaped directory under `/runpod-volume/huggingface-cache/hub/`) → a plain flat
  directory, `HUNYUAN_I2V_DIR`, staged by the shared fetch job and read with
  `local_files_only=True`. `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` are now set — the old
  worker deliberately did *not* set them because Model Caching still needed one resolvable Hub
  call to locate its own cache snapshot.
- **`build_check.py` added** (new file): fails the image build on CPU if torch was swapped for a
  CUDA 13 wheel, or if `HunyuanVideoImageToVideoPipeline` / `HunyuanVideoTransformer3DModel` lost
  the kwargs this handler passes them. No GPU is touched.
- **`num_frames` default**: 121 → 129 (see Handler contract above).
- Everything else — the fit/crop helper, the ffmpeg mux, the bf16-transformer/fp16-rest split, VAE
  tiling, the OOM message — is unchanged from the worker that was already running on RunPod.

## Build

```bash
MSYS_NO_PATHCONV=1 PYTHONIOENCODING=utf-8 "/c/Program Files/Microsoft SDKs/Azure/CLI2/python.exe" -IBm azure.cli acr build \
  -r azswmcacr -t hunyuan-i2v-worker:<tag> \
  "workers/vid-hunyuan-i2v" \
  --no-logs --timeout 7200 --query runId -o tsv
```

`<tag>` = the short git sha of this repo's HEAD.

## Run

Through the Azure runner shim (`gpu/azure_runner.py`, applied on top of this image by
`gpu/Dockerfile.layer`), which imports this file's `handler.py` and feeds it jobs from Storage
Queue `gpu-hunyuan-i2v`. Not run directly — `python handler.py` tries to import `runpod`'s
serverless loop with no queue behind it.

## Files

| file | what it is |
|---|---|
| `Dockerfile` | the image; built on ACR as `azswmcacr.azurecr.io/hunyuan-i2v-worker` |
| `handler.py` | the serverless worker: boot-time weight check + pipeline load, per-job image fit + generate + encode |
| `safe_fetch.py` | the SSRF guard for `image_url`/`https://` job inputs — refuses non-public addresses and redirects |
| `build_check.py` | build-time checks: every import, the diffusers pipeline/transformer API shape, the SSRF guard |
