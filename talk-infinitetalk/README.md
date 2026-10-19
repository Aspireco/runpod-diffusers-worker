# talk-infinitetalk — InfiniteTalk talking presenter (A100 80GB)

A photo + an audio track -> a lip-synced talking video. An existing video + a new audio track -> a
re-dubbed talking video. Calls the official **MeiGen-AI/InfiniteTalk** generation code directly
(`wan.InfiniteTalkPipeline.generate_infinitetalk`), vendored into the image at a pinned commit, rather
than reimplementing any of its model math. This is a clean-room worker: it does not touch and does not
copy code from the existing `workers/avatar-infinitetalk/` (an older ComfyUI-based worker vendored from
an unlicensed third-party hub image, `wlsdml1114` — see `docs/LANE-AUDIT-AND-PLAN.md`, "UNTESTED —
third-party hub image... one mention in 3,468 lines of decisions").

## Premade search

| Looked for | Found | Used / rejected |
|---|---|---|
| An official generation code path, so the lane does not hand-roll audio-conditioned video diffusion | `github.com/MeiGen-AI/InfiniteTalk`, commit `50aa0a94184315407a991ae804d9b58d6d311ba8` (pinned 2026-10-17, HEAD of `main`, cross-checked with `git ls-remote`) | **Used.** `wan.InfiniteTalkPipeline` (in `wan/multitalk.py`) is called directly; `handler.py` only reproduces the thin per-job glue `generate_infinitetalk.py` does inline (there is no exported `prepare_audio()` function to import), restructured so the ~45GB of models load once instead of once per CLI call. |
| A ComfyUI-based alternative (the pattern used by the existing `avatar-infinitetalk` worker) | the same `wlsdml1114` third-party ComfyUI nodes that worker already vendors | **Rejected.** Out of scope by the task's own hard boundary, and it is the unlicensed-provenance path this new lane exists to avoid. The official repo ships a native (non-ComfyUI) Python pipeline, which is the better premade fit. |
| A fast-sampling LoRA so the lane is not stuck at 40 steps | official README names two: `vrgamedevgirl84/Wan14BT2VFusioniX` (FusionX, 8 steps) and `Kijai/WanVideo_comfy` (a T2V-trained rank32 LoRA, cross-applied to this I2V model, "4 steps" per the README text, though its own worked example uses the FusionX recipe, not this one). Also found: `lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v`, an **I2V-specific** rank64 LoRA from the same `lightx2v` org whose Wan2.2 LoRA is already used by `workers/vid-wan-a14b`. | **Used the lightx2v org's own I2V rank64 LoRA**, not the two the README demos. Reasons: (1) it targets I2V directly rather than being a T2V LoRA applied cross-task, (2) first-party `lightx2v` org provenance matches the precedent already trusted in this repo, (3) its key names (`diffusion_model.<layer>.lora_down.weight` / `lora_up.weight` / `diff` / `diff_b`) were read directly from the safetensors header via an HTTP range request and matched byte-for-byte against what `wan/wan_lora.py`'s `WanLoraWrapper._apply_lora_weights` expects — verified compatible before committing to it, not assumed. Applied with the README's "FusioniX or Lightx2v" recipe (steps=8, text_guide_scale=1.0, audio_guide_scale=2.0, shift=2.0), since that is the only complete, explicit parameter recipe the README gives for this LoRA class. |
| flash-attn, since the spec says not to compile it on ACR | `wan/modules/attention.py`: `attention()` tries `flash_attn_interface` then `flash_attn`, and **falls back to `torch.nn.functional.scaled_dot_product_attention`** when both are absent (lines 13-23, 157-188 of the vendored file) | **Not installed.** Confirmed in the source rather than assumed. The one attention path with NO fallback is `xformers.ops.memory_efficient_attention` in `SingleStreamAttention`/`SingleStreamMutiAttention` (the audio cross-attention) — `xformers` is installed for that; flash-attn is not needed anywhere. |
| TTS (for a "no audio supplied" mode) | `kokoro/` (Kokoro-82M) ships in the repo, used only when `--audio_mode tts` | **Rejected / not vendored.** The handler contract always takes an explicit `audio` track; `kokoro/` and its extra deps (`misaki[en]`, etc.) are dropped to keep the image lean. |
| Scene segmentation / prompt-extend / Gradio demo | `wan/utils/segvideo.py` (`scenedetect`), `wan/utils/prompt_extend.py` (`dashscope`), `app.py` (`gradio`) | **Rejected / not vendored.** Grep-verified none of these are imported by `wan/__init__.py`, `wan/multitalk.py`, or anything else the handler touches — they are only reachable from `generate_infinitetalk.py`'s CLI (which this worker does not run) or `app.py` (not run either). |

## Licence table

| Component | Source | Licence | Verified |
|---|---|---|---|
| InfiniteTalk generation code | `github.com/MeiGen-AI/InfiniteTalk` @ `50aa0a9` | Apache-2.0 | `LICENSE.txt` fetched directly — standard Apache 2.0 text, no added clauses |
| Wan2.1-I2V-14B-480P (base DiT + VAE + T5 + CLIP) | `huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P` | Apache-2.0 | HF README frontmatter `license: apache-2.0` + full `LICENSE.txt` fetched |
| InfiniteTalk audio-conditioning checkpoint | `huggingface.co/MeiGen-AI/InfiniteTalk` (`single/infinitetalk.safetensors` only) | Apache-2.0 | Repo README: "The models in this repository are licensed under the Apache 2.0 License" |
| chinese-wav2vec2-base (audio encoder) | `huggingface.co/TencentGameMate/chinese-wav2vec2-base` | MIT | HF README frontmatter `license: mit` |
| lightx2v step-distill LoRA | `huggingface.co/lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v` (`loras/...rank64.safetensors` only) | Apache-2.0 | Repo README: same Apache-2.0 statement as InfiniteTalk's; `LICENSE.txt` present |
| xformers | PyPI `xformers` | BSD | PyPI classifier `License :: OSI Approved :: BSD License` |
| xfuser | PyPI `xfuser` (xDiT project) | Apache-2.0 | `github.com/xdit-project/xDiT` is Apache-2.0 licensed |
| optimum-quanto | PyPI `optimum-quanto` | Apache-2.0 | PyPI `license: Apache-2.0` |
| decord | PyPI `decord` | Apache-2.0 | PyPI `license: APACHE` |
| torch / transformers / diffusers / accelerate | PyPI | BSD / Apache-2.0 | well-established, not re-checked here |

No GPL/AGPL/non-commercial terms anywhere in this stack. Nothing triggered the "STOP and report" clause.

## File list under `INFINITETALK_DIR` (default `/runpod-volume/infinitetalk`)

One subfolder per fetched HF repo, matching `gpu/manifests/infinitetalk-*.tsv` exactly (these are the
destinations I expect the fetch jobs to write to):

| Manifest | HF repo | Destination folder | Files kept | Size |
|---|---|---|---|---|
| `infinitetalk-wan21-i2v-480p.tsv` | `Wan-AI/Wan2.1-I2V-14B-480P` | `wan21-i2v-14b-480p/` | `config.json`, 7 DiT shards, `Wan2.1_VAE.pth`, `models_t5_umt5-xxl-enc-bf16.pth`, `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth`, `google/umt5-xxl/*` (4 tokenizer files), `xlm-roberta-large/*` (4 tokenizer files) | 82,265,144,597 B = 76.62 GiB |
| `infinitetalk-ckpt.tsv` | `MeiGen-AI/InfiniteTalk` | `infinitetalk-ckpt/` | `single/infinitetalk.safetensors` only (not `multi/`, not `comfyui/`, not `quant_models/`) | 9,948,708,152 B = 9.27 GiB |
| `infinitetalk-wav2vec2.tsv` | `TencentGameMate/chinese-wav2vec2-base` | `wav2vec2/` | `config.json`, `preprocessor_config.json`, `pytorch_model.bin` (not the fairseq `.pt` checkpoint — unused by the `transformers`-based loader) | 380,263,948 B = 0.35 GiB |
| `infinitetalk-lightx2v-lora.tsv` | `lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v` | `lightx2v-lora/` | `loras/Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors` only (not the `distill_fp8`/`distill_int8`/`distill_models` full checkpoints) | 739,472,104 B = 0.69 GiB |
| **Total** | | | | **93,333,588,801 B = 86.92 GiB = 93.33 GB** |

All four manifests list selective files only, with exact byte sizes from `HfApi().model_info(repo,
files_metadata=True)`. The LoRA manifest is optional at runtime — the handler degrades to the plain
40-step model if it is missing (see `LIGHTNING` below) — but is expected to be fetched so the lane runs
at its documented default.

Every required path is checked at **handler import time** (`_require()` in `handler.py`); a missing file
raises `RuntimeError: <what> missing at <path>: run the fetch jobs for lane infinitetalk`, which fails the
job cleanly via the Azure runner's own `load()` error handling rather than crash-looping the replica.

## Handler contract

```json
// input
{
  "image": "data:image/png;base64,..." ,     // OR "video": "...", exactly one required
  "audio": "https://.../line.wav",           // required
  "prompt": "a person talking to the camera",// optional, default shown
  "size": "infinitetalk-480",                // optional: "infinitetalk-480" | "infinitetalk-720"
  "steps": 8,                                // optional: default 8 with the lightx2v LoRA, 40 without it
  "seed": -1,                                // optional: -1 = random
  "max_seconds": 60,                         // optional: refuses longer audio with an error
  "fps": 25                                  // optional, accepted but ignored -- see below
}
```

```json
// output
{
  "video_url": "data:video/mp4;base64,...",
  "duration_seconds": 10.08,
  "width": 640, "height": 640,
  "fps": 25.0,
  "frames": 252,
  "generation_seconds": 742.3,
  "steps": 8, "seed": 1234, "lora_active": true, "size": "infinitetalk-480"
}
```

On failure: `{"error": "<Type>: <message>"}` — the handler never raises.

**`fps` is accepted but not honoured as a free knob.** The official code hardcodes 25fps into the audio
embedding's time alignment (`get_embedding()`: `video_length = audio_duration * 25`) and into the muxing
step; changing it would silently desync lip-sync against the audio rather than just changing frame rate.
The output always reports the true native rate (25.0).

**Minimum audio length** is enforced: `(FRAME_NUM + 1) / 25 ≈ 3.3s`, because the official pipeline
silently drops (and then asserts on) any audio embedding shorter than one 81-frame clip bucket. Shorter
audio returns a clear error rather than the official code's `AssertionError`.

**Re-dub mode** (`video` input) keeps the source video's own camera movement — the official pipeline
walks the source video's frames forward as a motion reference at each generated chunk; a static photo has
no movement to reuse, so the same frame is used every chunk (both behaviours are the vendored code's own,
not something this handler implements).

## Premade search note on image handling

A static photo does **not** need converting to a fake video first (the repo ships a
`tools/convert_img_to_video.py` for a different purpose — extending very long single-image generations
past ~1 minute, not required here): `wan/utils/utils.py`'s `extract_specific_frames()` already falls back
to `Image.open(path)` for any non-video extension, returning the same image regardless of the requested
frame index. The handler relies on this directly rather than reimplementing it.

## VRAM and speed on an A100 80GB

**Resident VRAM, measured against the file sizes above, not benchmarked:** DiT ~28GB (bf16) + T5 text
encoder ~11.4GB (bf16) + CLIP image encoder ~4.8GB (fp16) + VAE ~0.5GB (fp32) ≈ **~45GB**, comfortably
inside 80GB with room for activations at either resolution (`infinitetalk-480` buckets to 640x640;
`infinitetalk-720` to 960x960 — the same pixel count as the 1280x720 the `vid-wan-a14b` lane already runs
successfully on this same card). `OFFLOAD_MODEL=0` (default) keeps everything resident for speed; the
official code's own `num_persistent_param_in_dit=0` low-VRAM path exists as a fallback knob (not wired
through this handler, since it should not be needed on an 80GB card) if VRAM pressure ever shows up.

**Speed and cost: not measured** (the task's hard boundary is no GPU jobs). What follows is a reasoned
estimate, not a benchmark, scaled from the one same-hardware, same-approach real number already in this
repo: `docs/runbook.md`'s measured Wan2.2-A14B figures on an A100 — a **warm** 81-frame clip costs about
CA$0.80 (≈294s) at 4 lightning steps with CFG off (1 forward pass/step through one active 14B expert = 4
total passes), and a **cold** one costs about CA$2.92-3.78 (≈1033-1393s, dominated by loading ~63-126GB of
weights at the share's measured 234 MB/s).

What is known with certainty (from the vendored source, not estimated):
- **Forward passes per 81-frame (3.24s) chunk:** with the lightx2v LoRA (`text_guide_scale=1.0`, so the
  audio-CFG-only branch runs), 8 steps x 2 passes/step = **16 passes**. Without it, 40 steps x 3 passes/step
  = **120 passes** — a **7.5x** difference, before any per-pass cost difference.
- **Chunks needed:** the first chunk covers 81 frames; each later chunk adds `81 - motion_frame(9) = 72`
  new frames (2.88s). A 10s clip needs **4 chunks**; a 60s clip needs **21 chunks**.

Scaling the wan-a14b anchor's per-pass cost (same 14B-active-parameter order of magnitude, one MoE expert
active per step there vs. this lane's single dense 14B DiT plus its added audio cross-attention) across
that many more passes and chunks puts a **warm** generation at very roughly **minutes per chunk** with the
LoRA and correspondingly longer without it — i.e. plausibly low-single-digit minutes for a 10s clip and
several times that for 60s, several times longer again without the LoRA. Treat this as an order-of-magnitude
planning number, not a quote. **Measure it properly with a real job** before relying on it — the
established pattern for that in this repo is `vid-wan-a14b/bench.py`'s fixed-input harness (see
`docs/SWITCHBOARD.md`, "Phase 3 — test the untested").

## Build status

Built successfully on ACR after fixing the two patches above (attempt 1 `ch1u` and attempt 2 `ch1w`
each found one of them; attempt 3 `ch20` succeeded). Pushed and live:

- `azswmcacr.azurecr.io/infinitetalk-worker:3f1aada-talk` and `:latest`
- digest `sha256:368712c281b81f9e78b5543c03eaf41d98eb6c350f0ab6c87c37e0eef6cdb7a8`
- 11,069,235,336 B = 10.3 GiB (no weights baked in — this is the base image plus every pip package and
  the vendored `wan`/`src` source only)
- run `ch20`: queued 21:34:00Z, succeeded 21:50:48Z (16m46s total; the push step alone was 587s for an
  ~11GB image)

## How to run

Requires the fetch jobs for the four manifests above to have filled `/runpod-volume/infinitetalk/...`
first (wiring the fetch jobs is John's side, per the task's ownership boundary — the manifests and
expected destinations are in the table above).

```bash
# local build (same image ACR builds — see the ownership boundary for why this task didn't commit or push)
docker build -t infinitetalk-worker:dev workers/talk-infinitetalk

# one job, assuming /runpod-volume is mounted locally for a smoke test
docker run --gpus all -v /path/to/runpod-volume:/runpod-volume infinitetalk-worker:dev
# then POST {"input": {"image": "...", "audio": "...", "max_seconds": 10}} the way azure_runner.py /
# runpod.serverless would -- this image's CMD still speaks the RunPod serverless protocol directly.
```

Env knobs: `INFINITETALK_DIR`, `CKPT_DIR`, `INFINITETALK_CKPT`, `WAV2VEC_DIR`, `LORA_PATH` (paths, all
default under `INFINITETALK_DIR`), `LIGHTNING` (default `1`, set `0` to force the plain 40-step model),
`OFFLOAD_MODEL` (default `0`).

**Two build-time source patches, neither a behaviour change — both found by real ACR build failures,
not by inspection:**
1. `wan/modules/t5.py`'s `T5EncoderModel.__init__` has `device=torch.cuda.current_device()` as a
   *default parameter value*, which Python evaluates the moment the class body executes — i.e. at
   `import wan` time. ACR build agents have no GPU at all, so the first build attempt (run `ch1u`)
   failed there with `RuntimeError: Found no NVIDIA driver...`, before `handler.py` ever ran. Patched to
   the literal `0` right after cloning: the only caller in this worker's path
   (`InfiniteTalkPipeline.__init__` in `wan/multitalk.py`) always passes `device=torch.device('cpu')`
   explicitly, so the default is never actually read at runtime, and `0` is exactly what
   `torch.cuda.current_device()` itself returns on any single-GPU machine — this worker's actual target
   hardware.
2. `wan/multitalk.py` has a dead `from inspect import ArgSpec` at its top. `ArgSpec` was removed from
   the stdlib `inspect` module in Python 3.11 (this base image ships 3.12); the second build attempt
   (run `ch1w`) failed there with `ImportError: cannot import name 'ArgSpec'`. Grep confirms the name is
   never referenced anywhere else in the file, so the line is deleted outright.

## Files

| file | what it is |
|---|---|
| `Dockerfile` | the image; clones InfiniteTalk at the pinned commit, installs the (corrected) dependency set, runs `build_check.py` |
| `handler.py` | the serverless worker: boots the pipeline once, then per job validates inputs, extracts one audio embedding, calls `InfiniteTalkPipeline.generate_infinitetalk`, muxes the result |
| `build_check.py` | build-time import check — fails the build, not a billed GPU, exercising the exact same import surface as `handler.py` |
| `README.md` | this file |
