# vid-minimax-h3 — MiniMax-H3

Text, or first/last-keyframe, to video **with** its own soundtrack generated in the same denoising pass —
one transformer, one packed sequence of text + conditioning + video latents + audio latents, no separate
vocoder and no audio post-hoc pass. Azure GPU fleet lane `minimax-h3`, Canada Central only.

## Premade search

| Need | Searched | Using | Why |
|---|---|---|---|
| The model and its generation code | The original MiniMax native checkpoint/repo vs. diffusers' Modular Diffusers integration (`MiniMaxH3Blocks` / `ModularPipeline`) | **diffusers' own Modular Diffusers blocks**, `ModularPipeline.from_pretrained(..., workflow="fl2va")` | MiniMax-H3 is integrated in diffusers as Modular Diffusers blocks *only* — there is no classic `DiffusionPipeline` half to pick between. This is the one and only officially supported recipe (https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_h3), already proven on the published `diffusers==0.41.0` PyPI wheel (not installed from git) and already the exact diffusers version this repo runs on `vid-ltx25`. Nothing here is written from scratch. |
| Weight staging on Azure | A manual rewrite of `modular_model_index.json`'s baked-in `"MiniMaxAI/MiniMax-H3"` repo ids vs. diffusers' own local-directory resolution | **diffusers' own mechanism**: `ModularPipeline.from_pretrained(MINIMAX_H3_DIR, ...)` — passing the local flat directory itself as `pretrained_model_name_or_path` — rewrites every component spec to that path the instant its subfolder already contains that component's files (`_is_local_component`, `diffusers/modular_pipelines/modular_pipeline.py`) | No new mechanism needed and no hand-maintained json patch to keep in sync with a future diffusers bump: this is diffusers' own documented behaviour for "a local copy of the repo... load the components whose files are present locally from the copy" (the comment is in diffusers' own source). Proven in `build_check.py`'s `offline` check on a decoy directory, and confirmed by running it locally against the real `diffusers==0.41.0` wheel before this was ever built on ACR. |
| House worker pattern | `workers/vid-hunyuan-i2v/{Dockerfile,handler.py,build_check.py,safe_fetch.py,README.md}` | **Copied the structure exactly**; `safe_fetch.py` copied byte-for-byte | Every other video worker in this repo (`vid-hunyuan-i2v`, `vid-wan-a14b`, `vid-ltx25`, ...) already solves "boot-time weight presence check, build-time class/offline/SSRF checks in a separate file, flat local directory, offline env vars" — reusing it rather than re-deriving it is the whole point of a house pattern. |
| diffusers/transformers versions | Installing diffusers from git at a pinned commit (what the model's own age would normally require) vs. the current PyPI release | **`diffusers==0.41.0`, `transformers==5.20.1`** — both already the exact pin this repo runs for `vid-ltx25` | MiniMax-H3 landed in diffusers main 2026-08-05 (PR #14355), got a VAE-decode fix 2026-09-28 (#14754) and single-file support 2026-10-01 (#14839); `diffusers==0.41.0` on PyPI postdates all three (confirmed by extracting the published wheel and finding `diffusers/modular_pipelines/minimax_h3/` inside it, not assumed from a changelog). Installing from git at a sha would only add a second, less-tested path for no benefit once the release already carries it. |

## Licence — read before production use, and before touching the registry this builds on

**The MiniMax H3 Community License's Territory excludes the European Union, the United Kingdom, South
Korea AND the United States.** This is the opposite shape to the Tencent Hunyuan Community License this
repo's other video lanes (`vid-hunyuan-i2v`, `vid-wan-animate`) carry, where the US is *inside* the grant
and only the EU/UK/South Korea are excluded. Mixing the two up is the single most expensive mistake
possible here.

| Component | Source | Licence | Territory / output restriction |
|---|---|---|---|
| MiniMax-H3 code path | `diffusers` (Apache-2.0), calling the model's own Modular Diffusers blocks | Apache-2.0 | n/a — generic pipeline/block code, not MiniMax's |
| MiniMax-H3 weights | `MiniMaxAI/MiniMax-H3` | **MiniMax H3 Community License** | Territory = worldwide **excluding the EU, UK, South Korea and the United States**. Confirmed by the task brief and the repo's own `LICENSE` file at the Hub. |
| Generated video + audio (Outputs) | — | Same licence | Binds *Outputs*, not just weights — generated video/audio may not be used, reproduced, distributed or displayed outside the Territory. "Powered by MiniMax H3" attribution is carried in this worker's own output `licence` field. |

**Why this image is built and pushed only to `azswgpuacrca` (Canada Central), never
`azswmcacr` (East US 2):** a build on a US registry puts the Works on US soil before a single job has run.
The lane this image serves is wired to run only in the Canada Central GPU environment, on Canada storage
(`azswgpuacrca`'s own `azswgpufilesca` share) — see `D:\claude projects\Azure setup\CLAUDE.md`'s Hunyuan
rule, which this worker follows for a different reason (Hunyuan is excluded from the EU/UK/Korea and
*allowed* in the US; this model is the reverse, excluded from the US too) but the same Canada-only
mechanism.

## Handler contract

Input (`job["input"]`):

```json
{
  "prompt": "a red fox trotting through a snowy pine forest, snow crunching underfoot",
  "image": "data:image/png;base64,... or raw base64 or https://... (optional first frame)",
  "last_image": "same formats (optional last frame)",
  "seconds": 5,
  "width": 1344,
  "height": 768,
  "steps": 30,
  "seed": -1
}
```

- `prompt` (required).
- `image` / `last_image` (optional): data URL, raw base64, or an `https://` URL, through
  `safe_fetch.fetch_image_bytes` — the same SSRF guard `vid-hunyuan-i2v` uses, copied in verbatim. `http://`
  is refused; redirects are refused, not followed.
- `seconds` (default **5**, clamped to **5..10**): `num_frames = round(seconds * 24)` is handed to the
  pipeline, which snaps it up to the next `17n + 5` the video VAE can decode (the model's own window is
  5-15s; this worker's own clamp is narrower, per the task brief).
- `width` / `height` (optional, both default to **1344x768** when either is given, clamped **512..1344**,
  rounded to the nearest multiple of 32): left out entirely, the pipeline derives its own canvas from the
  first keyframe's aspect ratio, or 16:9 with no keyframe at all.
- `steps` (default 30, clamped **1..50**): `num_inference_steps`. MiniMax-H3's two transformer partitions
  are **guidance-distilled** — there is no `guidance_scale` and no `negative_prompt` for either of them,
  and neither is accepted here.
- `seed`: -1 (default) picks a random seed; otherwise deterministic (one `torch.Generator`, used for the
  keyframe/reference noise, then the video noise, then the audio noise, in that order).

Output:

```json
{
  "video": "data:video/mp4;base64,...",
  "seconds": 5.17,
  "frames": 124,
  "width": null,
  "height": null,
  "seed": 1234567890,
  "generation_seconds": 180.4,
  "model": "MiniMaxAI/MiniMax-H3",
  "licence": "MiniMax H3 Community License (Territory excludes EU/UK/South Korea/US; binds Outputs; Powered by MiniMax H3)"
}
```

`width`/`height` are `null` whenever the request left both out and let the pipeline pick its own canvas —
they are not back-computed from the returned frames.

On failure: `{"error": "<Type>: <message>"}`, never a raised exception.

## Memory recipe (reasoned from diffusers' own docs — no GPU job was started)

Single A100 80GB replica, 220GiB host RAM (per the task brief):

```python
manager = ComponentsManager()
pipe = ModularPipeline.from_pretrained(MINIMAX_H3_DIR, workflow="fl2va", components_manager=manager,
                                       local_files_only=True)
pipe.load_components(workflow="fl2va", dtype=torch.bfloat16)
manager.enable_auto_cpu_offload(device="cuda", memory_reserve_margin="12GB")
```

The `transformer/` partition alone is 61.7GB in bf16 and the Qwen3-VL conditioner is another 62.1GB — the
two do not sit resident on one 80GB card together with the two VAEs and activations. `ComponentsManager` +
`enable_auto_cpu_offload` is diffusers' own documented recipe for exactly this card class: the weights live
in the replica's 220GiB of host RAM and the manager moves onto the GPU only what the current step needs.
`workflow="fl2va"` on **both** `from_pretrained` and `load_components` matters twice over: it is what keeps
`transformer_ref/` (the `ref2va` omni-reference partition, another 61.7GB) out of the component dict
entirely, and it is what this worker's manifest and fetch job assume is staged — see the next section.
`memory_reserve_margin="12GB"` is the exact figure the task brief gives; no GPU job ran to tune it further.

## Files under `$MINIMAX_H3_DIR` (default `/runpod-volume/minimax-h3`)

`gpu/manifests/minimax-h3.tsv` — **61 files, 144.05 GB** — is exactly what the `fl2va` workflow loads:
`model_index.json`, `modular_model_index.json`, and everything under `transformer/`, `text_encoder/`,
`tokenizer/`, `processor/`, `vae/`, `audio_vae/`, `scheduler/`, `audio_scheduler/`. **Deliberately
excluded**, because `fl2va` never touches them: `transformer_ref/` (the `ref2va`-only partition, another
61.7GB), `FL2VA/` and `Ref2VA/` (the repo's own native-checkpoint example dirs, not the diffusers layout),
`assets/`, `docs/`, `scripts/`. Built from the Hugging Face API tree
(`https://huggingface.co/api/models/MiniMaxAI/MiniMax-H3/tree/main?recursive=true`), not estimated.

```
$MINIMAX_H3_DIR/
├── model_index.json
├── modular_model_index.json
├── transformer/config.json, diffusion_pytorch_model-0000{1..14}-of-00014.safetensors, diffusion_pytorch_model.safetensors.index.json
├── text_encoder/{config,chat_template,generation_config,preprocessor_config,tokenizer_config,video_preprocessor_config,vocab}.json, merges.txt, tokenizer.json, model-0000{1..14}-of-00014.safetensors, model.safetensors.index.json
├── tokenizer/{tokenizer_config,vocab}.json, merges.txt, tokenizer.json
├── processor/{chat_template,preprocessor_config,tokenizer_config,video_preprocessor_config,vocab}.json, merges.txt, tokenizer.json
├── vae/config.json, diffusion_pytorch_model-0000{1..3}-of-00003.safetensors, diffusion_pytorch_model.safetensors.index.json
├── audio_vae/config.json, diffusion_pytorch_model.safetensors
├── scheduler/scheduler_config.json
└── audio_scheduler/scheduler_config.json
```

Wire the fetch job with repo id `MiniMaxAI/MiniMax-H3`, destination `$MINIMAX_H3_DIR`, against
`gpu/manifests/minimax-h3.tsv`.

## Build

**Canada registry only — see the Licence section above.**

```bash
MSYS_NO_PATHCONV=1 "/c/Program Files/Microsoft SDKs/Azure/CLI2/python.exe" -X utf8 -IBm azure.cli acr build \
  -r azswgpuacrca -t minimax-h3-worker:<short git sha> \
  -f vid-minimax-h3/Dockerfile vid-minimax-h3 \
  --no-logs --timeout 7200 --query runId -o tsv
```

`<short git sha>` = the short git sha of this repo's HEAD, taken at or after the commit that produced the
image being tagged (commit first, so the tag matches what was actually built).

## Run

Through the Azure runner shim (`gpu/azure_runner.py`, applied on top of this image by
`gpu/Dockerfile.layer`), which imports this file's `handler.py` and feeds it jobs from Storage Queue
`gpu-minimax-h3` in the Canada Central environment. Not run directly — `python handler.py` tries to import
`runpod`'s serverless loop with no queue behind it.

## Files

| file | what it is |
|---|---|
| `Dockerfile` | the image; built and pushed **only** on `azswgpuacrca.azurecr.io/minimax-h3-worker` (Canada Central) |
| `handler.py` | the serverless worker: boot-time weight presence check, `ModularPipeline` load (fl2va), per-job generate + `encode_video` mux |
| `safe_fetch.py` | the SSRF guard for `image`/`last_image` job inputs — copied verbatim from `vid-hunyuan-i2v` |
| `build_check.py` | build-time checks: every import (incl. the minimax_h3 reference classes and `Qwen3VLForConditionalGeneration`), the offline local-directory loading trick (proved on a decoy dir, no real weights), the SSRF guard |
