# vid-wan-animate — Wan2.2-Animate-14B (character animation + character replacement)

A character image + a driving video -> the character performs the video's motion ("animate", on
its own background) or replaces the person inside the video ("replace"). Azure GPU fleet lane
`wan-animate`, A100 80GB PCIe.

## Premade search

| Need | Searched | Using | Why |
|---|---|---|---|
| The model and its generation code | Wan-Video/Wan2.2 official repo, its diffusers port | **Official repo** (`github.com/Wan-Video/Wan2.2`, Apache-2.0), pinned commit `1ea34ff48f87168174e12956e200b1d908b1c5ff` (2026-09-21, latest at the time this was built) | Diffusers does not have a `WanAnimatePipeline` as of this build (checked its release notes and the pipeline list) -- animate's face/motion conditioning and preprocessing are not ported. The task also explicitly asked for the official code path. |
| Preprocessing (pose/mask extraction) | The same repo's `wan/modules/animate/preprocess/` (ProcessPipeline: Yolo detector + ViTPose + SAM2 + optional FLUX retargeting) | **Official preprocessing, as-is**, with one component swapped (below) | It is the code the model was trained against (pose conditioning format, face-crop convention, mask-dilation strategy). Reimplementing it would risk a format mismatch the model was never trained to see. |
| Person detector for preprocessing | Rejected the bundled one (AGPL-3.0, see licence table); considered RT-DETR via HF `transformers` (Apache-2.0, confirmed on its model card) vs. torchvision's own Faster R-CNN v2 (BSD-3-Clause, confirmed against `pytorch/vision`'s LICENSE) | **torchvision Faster R-CNN v2, COCO-pretrained** | Zero new pip dependency (torchvision is already pinned in every lane's base image) and zero new runtime dependency (RT-DETR would add `transformers`' object-detection image processor, which Wan2.2 doesn't otherwise need). Reshaped around the existing `pose2d.Yolo` interface with a small adapter (`detector_adapter.py`) rather than editing vendored code. |
| Enhanced pose retargeting for animate mode | `UserGuider.md`'s `--use_flux` path (FLUX.1-Kontext-dev image editing) | **Not implemented; `retarget_flag=True, use_flux=False`** (the "basic" retargeting path) | FLUX.1-Kontext-dev is a second multi-GB diffusion model with its own weights, and the official guide itself says the FLUX path "is NOT guaranteed to produce the expected results" and needs a human to inspect the intermediate output before proceeding -- not something a batch GPU job can do. Basic retargeting needs no extra weights and is still the officially recommended default over no retargeting at all. |
| Weight staging | Hugging Face `snapshot_download` (already the fetch job's own tool, `gpu/fetch/fetch_xet.py`) vs. hand-rolled sync | **The existing fetch job, unchanged** | Premade and already proven on five other lanes (wan-a14b, wan-t2v-a14b, zimage, flux-klein, cosmos3, voice-qwen3). |

Nothing here was built from scratch except `detector_adapter.py` (a ~130-line adapter, not a model)
and this README/manifest pair. The model, its weights, and its preprocessing are 100% the official
project's.

## The licence fix, in full

| Component | Source | Licence | Shipped? |
|---|---|---|---|
| Wan2.2 code | github.com/Wan-Video/Wan2.2 @ `1ea34ff4` | Apache-2.0 (repo `LICENSE.txt`, confirmed) | Yes, pip-installed from that commit |
| Wan2.2-Animate-14B weights | huggingface.co/Wan-AI/Wan2.2-Animate-14B | Apache-2.0 (HF `license` tag, confirmed via `HfApi().model_info`) | Yes, selective (manifest below) |
| T5 text encoder (`models_t5_umt5-xxl-enc-bf16.pth`, tokenizer `google/umt5-xxl/`) | same repo | Apache-2.0 (same repo licence) | Yes |
| CLIP (`models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth`, tokenizer `xlm-roberta-large/`) | same repo | Apache-2.0 (same repo licence) | Yes, selective -- see "tokenizer trim" below |
| Relighting LoRA (`relighting_lora.ckpt`) | same repo | Apache-2.0 (same repo licence) | Yes, one copy (the `.ckpt` the code actually loads, not the duplicate `relighting_lora/` PEFT-format folder also in the repo) |
| ViTPose wholebody pose model (`process_checkpoint/pose2d/`) | same repo, originally ViTAE-Transformer/ViTPose | Apache-2.0 (both the Animate-14B repo tag and `ViTAE-Transformer/ViTPose`'s own `LICENSE`, confirmed) | Yes, in full -- see "why the whole directory" below |
| SAM 2 (`process_checkpoint/sam2/sam2_hiera_large.pt`, pip package) | huggingface.co + `github.com/facebookresearch/sam2` @ `0e78a118` (the exact commit Wan2.2's own `requirements_animate.txt` pins) | Apache-2.0 (confirmed via `facebookresearch/sam2`'s `LICENSE`) | Yes, one checkpoint variant (`large`, the one `preprocess_data.py` hardcodes) -- tiny/small/base_plus and `download_ckpts.sh` are not fetched |
| **Person detector (`process_checkpoint/det/yolov10m.onnx`)** | Ultralytics YOLOv10, both `THU-MIG/yolov10` and `ultralytics/ultralytics`'s own integration | **AGPL-3.0** (confirmed against both repos' `LICENSE`, 2026-10-17) | **No. Never downloaded, never referenced.** |
| Replacement detector | torchvision (`fasterrcnn_resnet50_fpn_v2`, COCO-pretrained weights) | BSD-3-Clause (`pytorch/vision`'s `LICENSE`; weights are published and hosted by the same project under the same terms -- there is no separate per-weight licence file, which is the project's standing convention for every model it ships) | Yes, baked into the image (~170MB) |
| FLUX.1-Kontext-dev (enhanced retargeting) | black-forest-labs | Not used | No -- see premade-search table |

**How the AGPL detector was removed** (see `detector_adapter.py`'s module docstring for the full
version): `wan/modules/animate/preprocess/pose2d.py` builds its detector as a bare call,
`self.detector = Yolo(detector_checkpoint, device)`, inside `Pose2d.__init__`. Python resolves that
bare name `Yolo` from `pose2d`'s own module globals *every time `Pose2d.__init__` runs*, not at
class-definition time, so `handler.py` patches it before any `ProcessPipeline` is built:

```python
import pose2d                                  # the vendored module, added to sys.path
import detector_adapter
pose2d.Yolo = detector_adapter.PersonDetector  # now Pose2d.__init__'s Yolo(...) call builds this
```

`PersonDetector` presents the exact interface `Pose2d.__call__` relies on (`.preprocess(image) ->
(data, shape)`, then `detector(data[None], shape[None])[0][0]["bbox"]` for the best person's
`[x1,y1,x2,y2,score]`), backed by torchvision's Faster R-CNN v2 instead of the ONNX YOLOv10 model.
Nothing in `pose2d.py`, `pose2d_utils.py` or `process_pipepline.py` is edited. `build_check.py`'s
`detector` check exercises this patch at build time (CPU, a random frame, no weights needed beyond
the baked-in torchvision ones).

**Why the whole `pose2d/` directory ships, byte for byte identical to the source repo:** the file
named `vitpose_h_wholebody.onnx` is not a single file on the Hub -- it is a *directory* of ~394
files: one small ONNX graph (`end2end.onnx`) plus one file per weight tensor, stored as ONNX
"external data" (`onnx__MatMul_2804`, `backbone.blocks.0.norm1.weight`, etc.). `onnxruntime`
resolves every external-data reference relative to that directory at load time, so every one of
those ~394 files is required; there is nothing to selectively trim here the way there is for the
CLIP tokenizer below.

**Tokenizer trim (the "one format, not all five" the task flagged):** the repo's
`xlm-roberta-large/` folder ships the upstream `FacebookAI/xlm-roberta-large` tokenizer AND five
copies of its full pretrained model weights (`flax_model.msgpack`, `model.safetensors`,
`pytorch_model.bin`, `tf_model.h5`, `onnx/model.onnx*` -- about 11.2GB total). None of the five are
used: `wan/modules/animate/xlm_roberta.py` is a from-scratch `nn.Module` whose actual weights come
from the combined `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` checkpoint; the
`xlm-roberta-large/` folder is only ever opened for its tokenizer
(`transformers.AutoTokenizer.from_pretrained(tokenizer_path)`, confirmed by reading
`wan/modules/tokenizers.py`). I verified empirically (not just by reading the code) that
`config.json` + `sentencepiece.bpe.model` alone (4.8MB, no `tokenizer.json`, no
`tokenizer_config.json`, no `special_tokens_map.json`) load via `AutoTokenizer.from_pretrained` and
tokenize a test string to byte-identical ids as the full file set -- those two files are the
manifest's `xlm-roberta-large/*` entries.

## Known uncertainties and deliberate deviations (please read before trusting this blind)

- **Serving both modes from one resident process.** `generate.py` is a one-shot CLI: one process,
  one task, so `use_relighting_lora` is a constructor-time choice there. This worker serves a queue
  of jobs of *either* mode from one loaded model (loading a second 14B model to keep both modes
  resident would roughly double VRAM for no benefit). `handler.py` loads the relighting LoRA once at
  boot and toggles it per job with diffusers' `PeftAdapterMixin.enable_adapters()` /
  `disable_adapters()` on `ANIMATE.noise_model`. `build_check.py` confirms both methods exist on
  `WanAnimateModel` at build time. **I could not confirm on a GPU that toggling actually changes
  generated output as expected** -- the task's scope excludes starting GPU jobs, and nothing else in
  `wan/animate.py`'s `generate()` reads `use_relighting_lora` after construction, so toggling the
  adapter looks complete from reading the code, but this is the one piece of this worker I would
  check first against a real "animate" job if it ever comes back looking subtly lit like replace mode.
- **einops, moviepy and librosa are not in Wan2.2's own `requirements.txt` / `requirements_animate.txt`**,
  despite unconditional top-level imports reachable from a plain `import wan` (einops: `wan/animate.py`,
  `model_animate.py`, `face_blocks.py`, `motioner.py`, `auxi_blocks.py`, `s2v/*.py`; moviepy:
  `preprocess/process_pipepline.py`; librosa: `modules/s2v/audio_encoder.py`, pulled in because
  `wan/__init__.py` eagerly imports `speech2video`, even though this lane never calls it). Found by
  grepping every file the import chain actually touches, not by trusting the two requirements files;
  added explicitly in the Dockerfile.
- **`onnxruntime-gpu` replaces `requirements_animate.txt`'s plain `onnxruntime`.** `pose2d.py`
  explicitly requests `CUDAExecutionProvider`; the CPU-only PyPI package never compiles that provider
  in, so ViTPose would silently fall back to CPU for every frame instead of erroring -- a slow bug,
  not a loud one. `build_check.py`'s `imports` check asserts the provider is actually present.
- **`generate.py`'s own `--refert_num` CLI default is 77**, which contradicts its help text
  ("Recommended to be 1 or 5") and `WanAnimate.generate()`'s own
  `assert refert_num == 1 or refert_num == 5` -- using the CLI with no override would crash. Looks
  like a copy-paste of `--frame_num`'s default. `handler.py` uses `refert_num=5` (better cross-chunk
  continuity than 1, for a handful of extra overlap frames), not 77.
- **fps is fixed at the model's native 30, not exposed as a job parameter.** `generate.py` always
  encodes output at `cfg.sample_fps` (30) regardless of what fps preprocessing used; passing anything
  else to preprocessing would desync the output's nominal frame rate from how many frames were
  actually generated. The task's own input list doesn't include `fps`, which matches.
- **`resolution` only controls pixel *area*, not literal width x height.** The official preprocessing
  (`resize_by_area` -> `calculate_new_size`) derives the actual output width/height from the *driving
  video's own* aspect ratio and `width*height` as a target area -- `"1280x720"` and `"720x1280"` are
  equivalent inputs. Confirmed by reading `wan/modules/animate/preprocess/utils.py` directly.
- **sam2's CUDA extension (`connected_components.cu`) must compile at image build time** -- its
  `setup.py` (pinned commit) has no env-var skip, and it is not dead code:
  `sam_utils.py`'s `build_sam2_video_predictor()` sets `++model.fill_hole_area=8` by default, which
  calls the compiled op in "replace" mode. `TORCH_CUDA_ARCH_LIST=8.0` keeps the compile to one
  architecture (A100, sm_80). Confirmed nvcc is present in the base image (see Dockerfile's flash-attn
  comment, same probe build).
- **flash-attn is a hard dependency, confirmed, not assumed.** `clip.py` and `model_animate.py` call
  `flash_attn.flash_attn_varlen_func` directly, bypassing `wan/modules/attention.py`'s `attention()`
  dispatcher (the one with a torch-SDPA fallback) that the base Wan t2v/i2v model uses. A prebuilt
  wheel is used instead of compiling from source (hours); the exact build needed -- Python 3.12
  (cp312), torch 2.8, CUDA 12, cxx11 ABI **True** -- was confirmed by probing the base image directly
  (not assumed): ACR run `ch1p`, registry `azswmcacr`, throwaway tag `wan-animate-probe:tmp`, 2026-10-17.
- **`import wan` cannot run on a GPU-less machine, found live during the build (ACR run `ch1x`).**
  `wan/modules/t5.py`'s `T5EncoderModel.__init__` writes `device=torch.cuda.current_device()` as a
  default argument value, which Python evaluates at class-definition time -- i.e. the moment
  `import wan` reaches that module, not when the class is later instantiated. ACR's build agents have
  no NVIDIA driver at all, so a plain `import wan` there raises `RuntimeError: Found no NVIDIA driver
  on your system`. This only affects `build_check.py`'s own build-time verification (it needs
  `import wan` to check the API shape); `_patch_cuda_current_device_for_cpu_build()` stubs
  `torch.cuda.current_device` before that import, CPU-build-time only. The real worker runs on an
  actual A100 at runtime, where the unpatched call succeeds normally -- `handler.py` needs no
  equivalent patch and has none.

## VRAM / time expectations on an A100 80GB (reasoned, not benchmarked -- no GPU job was started)

Architecture: Wan-Animate's noise model is a **single dense 14B transformer** (not the two-expert
MoE the plain i2v/t2v A14B models use -- confirmed from `wan/configs/wan_animate_14B.py`, which has
one `dim`/`num_layers` set, no `transformer_2`), plus a CLIP image/text tower, a T5 text encoder, a
VAE, and (replace mode) a rank-128 relighting LoRA. `generate()` has no mid-run CPU offload of its
own (unlike `WanI2V.generate()`); everything it uses must be GPU-resident except what the
constructor already routes to CPU (`t5_cpu=True` here, saving ~11GB since the prompt is tiny and
encoded once).

| | estimate | basis |
|---|---|---|
| VRAM, animate mode | ~40-55GB | noise model bf16 ~28GB + CLIP ~5GB + VAE (fp32) ~1-2GB + flash-attention activations for a 77-frame/720p chunk (~75k "tokens", linear in flash-attn, not quadratic) + ViTPose/detector overhead |
| VRAM, replace mode | ~45-62GB | as above + relighting LoRA (~1-3GB resident) + SAM2-large (~1-2GB resident) |
| Generation, 5s clip (150 frames @ 30fps, ~2 chunks of 77) | ~4-10 min | 20 steps (default), `guide_scale=1.0` = no CFG = 1 forward pass/step, single dense 14B expert -- roughly a quarter of vid-wan-a14b's per-step cost (two 14B experts, CFG on), scaled by chunk count |
| Generation, 10s clip (300 frames, ~4 chunks) | ~8-20 min | linear scaling from the above |
| Preprocessing (both modes) | ~15-60s | per-frame Faster R-CNN + ViTPose, A100-fast for a few hundred frames |
| Preprocessing, replace mode only (SAM2 video propagation) | +30-90s | one video-level SAM2 pass over the whole clip |

Both fit comfortably under 80GB on this reasoning, with headroom -- consistent with `generate()`
never implementing offload itself (it would have to, if it didn't fit). **Treat the generation-time
range as a reasoned order-of-magnitude, not a measurement**; the task scope excludes starting a GPU
job to confirm it.

## Handler contract

Input (`job["input"]`):

```json
{
  "image": "data:image/png;base64,... or https://...",
  "video": "data:video/mp4;base64,... or https://...",
  "mode": "animate",
  "prompt": "optional, defaults to the model's own default prompt",
  "seed": -1,
  "steps": 20,
  "max_seconds": 10,
  "trim": false,
  "resolution": "1280x720"
}
```

- `image`/`video`: data URL or https URL, either accepted.
- `mode`: `"animate"` (character performs the video's motion on its own background, with pose
  retargeting) or `"replace"` (character replaces the person in the video, with the relighting LoRA
  and SAM2 mask propagation; no retargeting, per the official default).
- `max_seconds` (default 10): the driving video is refused with an error if longer, unless
  `trim: true`, which uses only the first `max_seconds`.
- `resolution` (optional, `"WxH"`): a target pixel area, not literal dimensions -- see above.
- `prompt`, `seed`, `steps`, `guide_scale`: all optional, default to the official `animate-14B`
  config's own values.

Output:

```json
{
  "video_url": "data:video/mp4;base64,...",
  "duration_seconds": 5.03,
  "width": 1280,
  "height": 720,
  "fps": 30.0,
  "frames": 151,
  "mode": "animate",
  "generation_seconds": 312.4
}
```

On failure: `{"error": "<Type>: <message>"}`, never a raised exception (matching every other lane
in this repo).

## Files the handler expects under `$WAN_ANIMATE_DIR` (default `/runpod-volume/wan-animate`)

Exactly the HF repo `Wan-AI/Wan2.2-Animate-14B`'s own layout, unmodified, minus what's excluded
above -- `gpu/manifests/wan-animate.tsv` (409 files, 53.77 GB) is everything "animate" mode needs on
disk; `gpu/manifests/wan-animate-replace.tsv` (2 files, 3.77 GB) adds what "replace" mode also
needs. Both manifests target the same destination directory (one `snapshot_download` call per
manifest, same `local_dir`). **Total: 411 files, 57.54 GB (53.59 GiB).**

**Both manifests must be fetched before this worker will boot at all**, even for an animate-only
deployment: `handler.py` loads the relighting LoRA once at boot and toggles it per job (see "Known
uncertainties" above) rather than loading a second model, so `relighting_lora.ckpt` and
`sam2_hiera_large.pt` are on the boot-time required-files list unconditionally, not just when a
"replace" job actually arrives.

```
$WAN_ANIMATE_DIR/
├── config.json, diffusion_pytorch_model-0000{1..4}-of-00004.safetensors,
│   diffusion_pytorch_model.safetensors.index.json      # the noise model (WanAnimateModel.from_pretrained)
├── Wan2.1_VAE.pth                                       # VAE
├── models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth   # CLIP
├── models_t5_umt5-xxl-enc-bf16.pth                      # T5 text encoder
├── relighting_lora.ckpt                                 # replace mode only
├── google/umt5-xxl/{special_tokens_map,tokenizer,tokenizer_config}.json, spiece.model   # T5 tokenizer
├── xlm-roberta-large/{config.json, sentencepiece.bpe.model}   # CLIP tokenizer (trimmed, see above)
└── process_checkpoint/
    ├── pose2d/vitpose_h_wholebody.onnx/   # end2end.onnx + ~393 external-data tensor files (in full)
    └── sam2/sam2_hiera_large.pt           # replace mode only
    # NOTE: no det/ -- the AGPL yolov10m.onnx is never fetched; see the licence section above.
```

Wire the fetch job with repo id `Wan-AI/Wan2.2-Animate-14B` and destination `$WAN_ANIMATE_DIR`
against both manifests (same destination for both calls to `fetch_xet.py`).

## Build

```bash
MSYS_NO_PATHCONV=1 PYTHONIOENCODING=utf-8 "/c/Program Files/Microsoft SDKs/Azure/CLI2/python.exe" -IBm azure.cli acr build \
  -r azswmcacr -t wan-animate-worker:<tag> -t wan-animate-worker:latest \
  -f "workers/vid-wan-animate/Dockerfile" "workers/vid-wan-animate" \
  --no-logs --timeout 7200 --query runId -o tsv
```

`<tag>` = the short git sha of this repo's HEAD plus `-animate` (e.g. `e06838a-animate`).

## Run

Through the Azure runner shim (`gpu/azure_runner.py`), which imports this file's `handler.py` and
feeds it jobs from Storage Queue `gpu-wan-animate` -- the same convention every other lane in
`gpu/manifests/` uses. Not run directly; `python handler.py` would try to import `runpod`'s
serverless loop with no queue behind it.

## Files

| file | what it is |
|---|---|
| `Dockerfile` | the image; built on ACR as `azswmcacr.azurecr.io/wan-animate-worker` |
| `handler.py` | the serverless worker: boot-time model/pipeline load, per-job preprocess + generate + encode |
| `detector_adapter.py` | the AGPL-detector replacement (torchvision Faster R-CNN v2), monkey-patched over `pose2d.Yolo` |
| `build_check.py` | build-time checks: every import, the `wan`/`WanAnimate` API shape, the detector patch, sam2's CUDA extension |
