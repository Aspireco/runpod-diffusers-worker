# lipsync-musetalk worker

MuseTalk 1.5 (TMElyralab, MIT) lip-sync for the Azure GPU fleet lane `musetalk` (NVIDIA T4,
16 GB). Re-syncs an existing video's mouth to new audio. Runs MuseTalk's own library code
directly (not a `scripts/inference.py` subprocess per job -- see "Why not call the CLI script"
below) against weights staged on the Azure Files share, with the models loaded once per replica.

## Premade search

- **Used:** the official repo, [`TMElyralab/MuseTalk`](https://github.com/TMElyralab/MuseTalk)
  (MIT), pinned at commit `0a89dec45a0192b824e3cf4daf96c239440c5ed8` (2025-09-26, HEAD of `main`
  at the time this image was built). This is the only maintained, MIT-licensed, open-weights
  lip-sync model with a v1.5 release; it is also the thing the task spec named directly. No
  other candidate was seriously in the running: Wav2Lip (academic-only licence on the original
  weights), SadTalker (non-commercial CC BY-NC-SA on its own weights), and most "realtime
  talking head" repos on GitHub either vendor Wav2Lip's weights transitively or have no licence
  file at all. MuseTalk is also already a known quantity in this fleet's neighbourhood --
  `workers/avatar-infinitetalk` runs a related (image-to-video) talking-head model on the same
  Azure setup -- but InfiniteTalk re-animates a still photo from a driving video, a different job
  from this lane's "re-sync this video's existing mouth to new audio," so it was not a fit to
  reuse directly.
- **Rejected:** reimplementing any part of the model. The handler calls MuseTalk's own functions
  (`load_all_model`, `get_landmark_and_bbox`, `datagen`, `get_image`, the VAE/UNet forward pass)
  for every frame of actual ML work; see "Why not call the CLI script" for the one structural
  change (lifting the one-time setup out of `main()` so it runs once per replica, not once per
  job) and `handler.py`'s module docstring for the rest.

## Why not call the CLI script per job

`scripts/inference.py`'s `main(args)` loads every model -- VAE, UNet, Whisper, the face parser,
and (via importing `musetalk.utils.preprocessing`, which constructs them at module scope) DWPose
and the S3FD face detector -- fresh on every invocation. That's correct for a one-shot CLI call
but wrong for a worker that should load once and drain a whole queue of jobs per replica (see
`gpu/azure_runner.py`: a replica keeps calling the handler until its queue is empty, then exits).
`handler.py` splits `main()` at the exact point upstream's own comments mark "Process each task":
everything before that runs once at import (this is boot steps 1-4 in the file), and everything
from "Process each task" onward runs per job, as `_lip_sync()`, against the already-loaded
globals. No math, model class, or algorithm is reimplemented -- same function calls, same
arguments, same v1.5 defaults -- only the CLI/argparse/multi-task-yaml shell around them is
replaced with a handler function. The three deliberate behavioural differences from upstream
(all commented inline in `handler.py`) are:
1. `bbox_shift` is honoured for v1.5 too. Upstream's `main()` hardcodes `bbox_shift = 0` for
   `version == "v15"` regardless of what was passed in; since this lane's contract takes
   `bbox_shift` as a parameter (default 0, matching upstream's own default), the caller's value
   is used instead of being silently discarded.
2. A face-less video returns `{"error": "no face detected in the source video"}` instead of
   reaching `datagen`'s `% len(vae_encode_latents)` with an empty list and raising
   `ZeroDivisionError` inside upstream's bare `except Exception: print(...)` (which just logs and
   moves to the next yaml task -- there is no "next task" in a single-job handler).
3. The final audio mux is this lane's own contract (`-c:v copy -c:a aac -movflags +faststart`)
   rather than upstream's bare `ffmpeg -i audio -i video out.mp4`, which leaves the audio codec
   to ffmpeg's defaults.

One pre-existing upstream quirk is deliberately *not* fixed, and is commented in `_lip_sync()`
where it occurs: `frame_list_cycle`/`coord_list_cycle` are built from the full, unfiltered frame
list (including a placeholder for any frame with no detected face), while
`input_latent_list_cycle` is built from the filtered list. If *some but not all* frames of a
video lack a detected face, the two cycles end up different lengths and can drift out of step.
This is upstream's own behaviour in `scripts/inference.py`'s `main()`, unchanged here; a video
with no detected face in *any* frame is caught cleanly by difference #2 above instead.

## Input / output contract

```
POST input:
{
  "video": "data:video/mp4;base64,..." | "https://...",   // required
  "audio": "data:audio/wav;base64,..." | "https://...",   // required
  "bbox_shift": 0,        // optional int, default 0
  "fps": 25,              // optional number; default = source video's fps, capped at 30 either way
  "max_seconds": 120      // optional number; longer audio is refused before any GPU work (hard cap 600)
}

Success:
{
  "video_url": "data:video/mp4;base64,...",
  "duration_seconds": 6.24,
  "width": 1280,
  "height": 720,
  "fps": 25.0,
  "generation_seconds": 41.8
}

Failure (always a 200 with this body, never a raised exception):
{ "error": "<ExceptionType>: <message>" }
```

If the audio is longer than the source video, MuseTalk's own ping-pong cycling
(forward frames, then reversed, repeated via modulo indexing) covers the extra length. If the
audio is shorter, the output is naturally trimmed to the audio -- the generated frame count is
driven by the audio's Whisper chunks, never by the source video's own length. Both are upstream's
existing mechanism; nothing extra was added for this.

## Files this handler expects under `$MUSETALK_DIR` (default `/runpod-volume/musetalk`)

| Subfolder | Files | From (HF repo) | Manifest |
|---|---|---|---|
| `musetalkV15/` | `unet.pth`, `musetalk.json` | `TMElyralab/MuseTalk` (only the `musetalkV15/` path) | `gpu/manifests/musetalk-musetalkv15.tsv` |
| `sd-vae/` | `config.json`, `diffusion_pytorch_model.bin` | `stabilityai/sd-vae-ft-mse` | `gpu/manifests/musetalk-sd-vae.tsv` |
| `whisper/` | `config.json`, `preprocessor_config.json`, `pytorch_model.bin` | `openai/whisper-tiny` | `gpu/manifests/musetalk-whisper.tsv` |
| `dwpose/` | `dw-ll_ucoco_384.pth` | `yzd-v/DWPose` | `gpu/manifests/musetalk-dwpose.tsv` |
| `face-parse-bisent/` | `79999_iter.pth`, `resnet18-5c106cde.pth` | `ManyOtherFunctions/face-parse-bisent` | `gpu/manifests/musetalk-face-parse-bisent.tsv` |

Total ≈ 4.39 GB. For the `TMElyralab/MuseTalk` repo specifically, the fetch job's **destination
directory must be `$MUSETALK_DIR` itself** (not `$MUSETALK_DIR/musetalkV15`): the manifest's path
column already includes the `musetalkV15/` prefix, matching that repo's own internal layout
(it also contains a sibling `musetalk/` folder for v1.0, which this lane does not fetch), so
`snapshot_download`'s own path-preserving behaviour lands the files at
`$MUSETALK_DIR/musetalkV15/...` correctly. The other four fetch jobs point their destination
directly at the named subfolder (e.g. `$MUSETALK_DIR/sd-vae`), since those repos keep their files
at repo root.

Boot fails fast and names the exact missing file plus "run the fetch job for lane musetalk" if
anything is absent -- see `handler.py`'s boot step 1.

Folder names mostly match upstream's own hardcoded layout (`musetalkV15`, `dwpose`,
`face-parse-bisent` -- including upstream's own "bisent" spelling, not "bisenet" -- are literal
relative-path strings baked into MuseTalk's code, see "How the models/ layout is built" below);
`sd-vae` and `whisper` are this lane's own choice, since those two are passed as plain call
arguments upstream, not hardcoded paths.

## How the `models/` layout is built (and why only two symlinks)

Traced against the actual MuseTalk source rather than assumed. Two places hardcode a
*CWD-relative* path with no way to override it from outside:
- `musetalk/utils/preprocessing.py` sets `config_file = './musetalk/utils/dwpose/<name>.py'` and
  `checkpoint_file = './models/dwpose/dw-ll_ucoco_384.pth'` as bare literals at module scope
  (this is also where DWPose's `init_model()` and the S3FD `FaceAlignment()` actually load, which
  is why importing this one module is what makes boot slow).
- `musetalk/utils/face_parsing/__init__.py`'s `FaceParsing.model_init()` defaults to
  `'./models/face-parse-bisent/resnet18-5c106cde.pth'` and `'.../79999_iter.pth'`, and
  `FaceParsing.__init__` never passes anything else in.

Both resolve against the process's current working directory, so `handler.py` builds exactly
this much under `/tmp/musetalk_run` (never under `/runpod-volume`, which stays read-only from
this worker's point of view) and `chdir`s there before importing anything from `musetalk`:
a symlink `musetalk -> /opt/MuseTalk/musetalk` (the vendored code, baked into the image) and
`models/dwpose` / `models/face-parse-bisent` symlinked to the real weight folders on the Azure
Files share. `musetalkV15`, `sd-vae` and `whisper` are **not** symlinked -- `load_all_model()`,
`AudioProcessor()` and `WhisperModel.from_pretrained()` all take their path as a normal call
argument, so `handler.py` just passes the real absolute path directly (including a small
`os.path.join("models", vae_type)`-defeating trick for `vae_type`, commented where it's used).

One more thing this relies on that is worth naming: `musetalk/utils/preprocessing.py` does
`from face_detection import FaceAlignment, LandmarksType` as a *bare top-level* import, which
only resolves because `musetalk/utils/__init__.py` appends its own directory onto `sys.path` the
moment `musetalk.utils` is imported -- a trick that works through the symlink unmodified, since
it computes the path from `__file__`, not from the CWD. No image changes were needed for this.

## Licence table (checked at source, not assumed)

| Component | Source | Licence | Checked |
|---|---|---|---|
| MuseTalk code + `musetalkV15/unet.pth`, `musetalk.json` | [`TMElyralab/MuseTalk`](https://github.com/TMElyralab/MuseTalk) | **MIT** (explicit: "no limitation for both academic and commercial usage") | Repo `LICENSE` file, fetched and read in full |
| sd-vae-ft-mse weights | [`stabilityai/sd-vae-ft-mse`](https://huggingface.co/stabilityai/sd-vae-ft-mse) | **MIT** | HF model card metadata + MuseTalk's own `LICENSE` third-party list (agree) |
| Whisper code (vendored, unused by this handler's path) + `openai/whisper-tiny` weights | [`openai/whisper`](https://github.com/openai/whisper) code; [`openai/whisper-tiny`](https://huggingface.co/openai/whisper-tiny) weights | Code **MIT** (GitHub `LICENSE`, Copyright OpenAI); HF model card tags the weights **Apache-2.0**. Either way, unrestricted commercial use | GitHub `LICENSE` fetched in full; HF model card checked; MuseTalk's own `LICENSE` lists it MIT |
| face-parsing.PyTorch code (vendored BiSeNet) | [`zllrunning/face-parsing.PyTorch`](https://github.com/zllrunning/face-parsing.PyTorch) | **MIT** | MuseTalk's own `LICENSE` third-party list; repo README confirms MIT |
| `79999_iter.pth`, `resnet18-5c106cde.pth` weights (fetched from the HF mirror, see below) | [`ManyOtherFunctions/face-parse-bisent`](https://huggingface.co/ManyOtherFunctions/face-parse-bisent) | HF repo tagged **WTFPL** (the mirror's own redistribution terms); the weights originate with `zllrunning/face-parsing.PyTorch` above (MIT) and `resnet18-5c106cde.pth` is byte-identical (same size, 46,827,520 bytes) to torchvision's own `https://download.pytorch.org/models/resnet18-5c106cde.pth` (**BSD-3-Clause**, PyTorch). Every applicable licence here permits unrestricted commercial use | HF model card checked; byte-size cross-check against `download.pytorch.org` run directly |
| DWPose weights (`dw-ll_ucoco_384.pth`) | [`yzd-v/DWPose`](https://huggingface.co/yzd-v/DWPose) | **Apache-2.0** | HF model card metadata; matches MuseTalk's own `LICENSE` third-party list exactly |
| S3FD face detector code + weight (vendored, `musetalk/utils/face_detection/`) | [`1adrianb/face-alignment`](https://github.com/1adrianb/face-alignment) | **BSD-3-Clause** (Copyright 2017, Adrian Bulat) | GitHub `LICENSE` fetched in full; matches MuseTalk's own `LICENSE` third-party list. (MuseTalk's `LICENSE` also separately credits the S3FD *architecture* via `yxlijun/S3FD.pytorch` with no licence text of its own -- the code and weight actually vendored and run here are the BSD-3-Clause face-alignment copy, confirmed by diffing the vendored files' structure against the upstream repo) |
| mmcv / mmdet / mmpose / mmengine | OpenMMLab | **Apache-2.0** (standard for the mmlab ecosystem) | Well-established; not independently re-fetched given the above five were the ones in question |
| diffusers, transformers, accelerate, huggingface_hub, librosa, einops, omegaconf, soundfile, opencv-python, runpod | PyPI | Apache-2.0 / BSD / MIT (standard, permissive) | Not independently re-fetched; none of this set has ever carried a restrictive licence |

**Nothing here is GPL, AGPL, or non-commercial.** No required piece triggered the "stop and
report" condition.

### Pip packages intentionally left out of `requirements.txt`

Installed MINUS `tensorflow==2.12.0`, `tensorboard==2.12.0` (a training/eval-only metric, not
imported anywhere on `scripts/inference.py`'s call path), `gradio==5.24.0` (only `app.py`'s demo
UI imports it), and `gdown`/`requests`/`imageio[ffmpeg]`/`ffmpeg-python`/`moviepy` (upstream's own
`scripts/inference.py` shells out to the `ffmpeg` binary directly via `subprocess`/`os.system`
rather than using any of these; `gdown`/`requests` are weight-download helpers, superseded here by
the fleet's own fetch-job pipeline). Every file this handler imports was traced import-by-import
first; `build_check.py` then proves the result actually imports inside the real image. This was a
size/build-time call (the task's "keep the image lean" instruction), not a licence one.

## VRAM and speed on a T4 (estimated, not benchmarked -- no GPU job was started for this build,
per this task's ownership boundary)

Rough resident-weight budget at the default `DTYPE=float16`: UNet ~1.7 GB, VAE ~0.17 GB, Whisper
encoder ~0.08 GB (all three cast to fp16), plus DWPose (~0.4 GB), the S3FD detector (~0.09 GB) and
the BiSeNet face parser (~0.1 GB) which upstream never casts down from fp32. That totals roughly
2.5 GB of weights, comfortably inside the T4's 16 GB alongside batch activations and the CUDA/
mmcv/mmpose runtime overhead.

The likely throughput bottleneck is not the UNet denoising batch (`batch_size=8`, run on the
GPU) but the landmark/face-detection pass: `get_landmark_and_bbox` hardcodes
`batch_size_fa = 1` in upstream's own code, i.e. every frame goes through DWPose + the S3FD
detector one at a time, with no batching, regardless of GPU class. Expect generation time to
scale with video length more through that per-frame pass than through the UNet step. This is an
inference from reading the code, not a measured number -- the first real benchmark will come
from an actual GPU job.

## Running it

Build (ACR; see the main runbook for the exact `az acr build` invocation this lane uses):

```
az acr build -r azswmcacr -t musetalk-worker:<tag> -t musetalk-worker:latest \
  -f workers/lipsync-musetalk/Dockerfile workers/lipsync-musetalk
```

Then layer the Azure runner on top (`gpu/Dockerfile.layer`) and queue a job with the input shape
under "Input / output contract" above. `MUSETALK_DIR` and `DTYPE` are the only two env vars this
handler reads; everything else about the GPU job queue, blob offload, and Storage Table status
is handled by `gpu/azure_runner.py`, unchanged.
