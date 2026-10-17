"""MuseTalk 1.5 lip-sync (TMElyralab/MuseTalk, MIT) for the Azure GPU fleet lane `musetalk` (T4, 16 GB).

Re-syncs an existing video's mouth to new audio. Calls MuseTalk's own library code (the same
functions `scripts/inference.py` calls: `load_all_model`, `get_landmark_and_bbox`, `datagen`,
`get_image`, the VAE/UNet forward pass) rather than reimplementing any of it. The one thing this
does NOT do is shell out to `scripts/inference.py` itself: that script's `main()` loads every
model (VAE, UNet, Whisper, face parser, and -- via `musetalk.utils.preprocessing`'s own
module-level code -- DWPose and the S3FD face detector) fresh on every call, which is fine for a
one-shot CLI but defeats a warm serverless worker that should load once and serve many queued
jobs (see azure_runner.py: a replica drains its whole queue before exiting). So the one-time
setup half of `main()` is lifted here to run once at import, and the per-task half (everything
from "Process each task" onward) runs per job against the already-loaded models. Nothing in
either half is reimplemented -- same functions, same call order, same defaults -- except where a
comment below says otherwise.

Input:  video (data: or https URL, required), audio (data: or https URL, required),
        bbox_shift (int, default 0), fps (default: source video's fps, capped 30),
        max_seconds (default 120; longer audio is refused before any GPU work).
Output: {"video_url": "data:video/mp4;base64,...", "duration_seconds", "width", "height",
         "fps", "generation_seconds"}.
Env:    MUSETALK_DIR (default /runpod-volume/musetalk) -- one subfolder per fetched HF repo:
        musetalkV15/{unet.pth,musetalk.json}, sd-vae/{config.json,diffusion_pytorch_model.bin},
        whisper/{config.json,preprocessor_config.json,pytorch_model.bin},
        dwpose/dw-ll_ucoco_384.pth, face-parse-bisent/{79999_iter.pth,resnet18-5c106cde.pth}.
        DTYPE (default float16 -- a T4 has no fast bf16; float32 also accepted).
"""

import base64
import copy
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

print("[boot] lipsync-musetalk handler starting", flush=True)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

print(f"[boot] torch {torch.__version__} cuda={torch.cuda.is_available()}", flush=True)

MUSETALK_REPO = "/opt/MuseTalk"          # the pinned git checkout, baked into the image
RUN_DIR = "/tmp/musetalk_run"            # writable scratch; MUSETALK_DIR (the Azure Files
MUSETALK_DIR = os.environ.get("MUSETALK_DIR", "/runpod-volume/musetalk")  # share) is never written to
DTYPE_NAME = os.environ.get("DTYPE", "float16")
USE_FP16 = DTYPE_NAME != "float32"
MAX_SECONDS_HARD_CAP = 600  # a caller's max_seconds is clamped here no matter what it asks for

# ---------------------------------------------------------------------------------------------
# Boot step 1: confirm every fetched file is actually there, with a message that names the fix.
# ---------------------------------------------------------------------------------------------
_REQUIRED = {
    "musetalkV15": ["unet.pth", "musetalk.json"],
    "sd-vae": ["config.json", "diffusion_pytorch_model.bin"],
    "whisper": ["config.json", "preprocessor_config.json", "pytorch_model.bin"],
    "dwpose": ["dw-ll_ucoco_384.pth"],
    "face-parse-bisent": ["79999_iter.pth", "resnet18-5c106cde.pth"],
}
for _sub, _files in _REQUIRED.items():
    for _fn in _files:
        _p = os.path.join(MUSETALK_DIR, _sub, _fn)
        if not os.path.isfile(_p):
            raise RuntimeError(f"missing {_p}: run the fetch job for lane musetalk ({_sub})")

# ---------------------------------------------------------------------------------------------
# Boot step 2: build the layout MuseTalk's own code expects, under /tmp, as symlinks only -- it
# never writes to /runpod-volume. Only two targets actually need this (traced against the real
# source): `musetalk/utils/preprocessing.py` hardcodes './musetalk/utils/dwpose/<config>.py' and
# './models/dwpose/dw-ll_ucoco_384.pth' as bare relative strings at MODULE scope (not parametrised
# at all), and `musetalk/utils/face_parsing/__init__.py`'s `FaceParsing.model_init()` hardcodes
# './models/face-parse-bisent/{79999_iter,resnet18-5c106cde}.pth' as default args that
# `FaceParsing.__init__` never overrides. Both resolve relative to the process's CWD, so CWD is
# set to RUN_DIR below. The musetalkV15/sd-vae/whisper weights are NOT symlinked: `load_all_model`,
# `AudioProcessor` and `WhisperModel.from_pretrained` all take their path as a normal call
# argument, so this handler just passes the real absolute path for those three directly (see
# "Boot step 4") instead of relying on any relative default.
# ---------------------------------------------------------------------------------------------
os.makedirs(os.path.join(RUN_DIR, "models"), exist_ok=True)
os.makedirs(os.path.join(RUN_DIR, "jobs"), exist_ok=True)  # tempfile.mkdtemp(dir=...) needs it to pre-exist
_musetalk_link = os.path.join(RUN_DIR, "musetalk")
if not os.path.islink(_musetalk_link):
    os.symlink(os.path.join(MUSETALK_REPO, "musetalk"), _musetalk_link)
for _sub in ("dwpose", "face-parse-bisent"):
    _link = os.path.join(RUN_DIR, "models", _sub)
    if not os.path.islink(_link):
        os.symlink(os.path.join(MUSETALK_DIR, _sub), _link)

os.chdir(RUN_DIR)
sys.path.insert(0, RUN_DIR)

# ---------------------------------------------------------------------------------------------
# Boot step 3: import MuseTalk's own code. Importing `musetalk.utils.preprocessing` is what
# actually constructs the DWPose model (mmpose `init_model`) and the S3FD face detector
# (`FaceAlignment`) -- at MODULE scope, as upstream wrote it, which is why this import happens
# exactly once here rather than per job.
# ---------------------------------------------------------------------------------------------
_t_models = time.time()
from musetalk.utils.utils import load_all_model, get_video_fps, datagen  # noqa: E402
from musetalk.utils.preprocessing import get_landmark_and_bbox, coord_placeholder  # noqa: E402
from musetalk.utils.blending import get_image  # noqa: E402
from musetalk.utils.face_parsing import FaceParsing  # noqa: E402
from musetalk.utils.audio_processor import AudioProcessor  # noqa: E402
from transformers import WhisperModel  # noqa: E402

print(f"[boot] dwpose + s3fd ready in {time.time() - _t_models:.1f}s", flush=True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------------------------
# Boot step 4: the three weights `load_all_model`/`AudioProcessor`/`WhisperModel` take as a call
# argument, passed as real absolute paths (no CWD-relative default involved). `vae_type` is not
# itself a path -- `load_all_model` builds it as `os.path.join("models", vae_type)` -- but
# `os.path.join` discards every earlier part the moment one argument is already absolute, so
# passing the real absolute sd-vae folder as `vae_type` lands the VAE exactly there.
# ---------------------------------------------------------------------------------------------
_t_unet = time.time()
VAE, UNET, PE = load_all_model(
    unet_model_path=os.path.join(MUSETALK_DIR, "musetalkV15", "unet.pth"),
    vae_type=os.path.join(MUSETALK_DIR, "sd-vae"),
    unet_config=os.path.join(MUSETALK_DIR, "musetalkV15", "musetalk.json"),
    device=DEVICE,
)
TIMESTEPS = torch.tensor([0], device=DEVICE)

if USE_FP16:
    PE = PE.half()
    VAE.vae = VAE.vae.half()
    UNET.model = UNET.model.half()
PE = PE.to(DEVICE)
VAE.vae = VAE.vae.to(DEVICE)
UNET.model = UNET.model.to(DEVICE)
WEIGHT_DTYPE = UNET.model.dtype

AUDIO_PROCESSOR = AudioProcessor(feature_extractor_path=os.path.join(MUSETALK_DIR, "whisper"))
WHISPER = WhisperModel.from_pretrained(os.path.join(MUSETALK_DIR, "whisper"))
WHISPER = WHISPER.to(device=DEVICE, dtype=WEIGHT_DTYPE).eval()
WHISPER.requires_grad_(False)

# v1.5 defaults from upstream's own scripts/inference.py argparse (left_cheek_width=90,
# right_cheek_width=90; FaceParsing's own class-level default of 80/80 is never used for v15).
FACE_PARSER = FaceParsing(left_cheek_width=90, right_cheek_width=90)

print(f"[boot] vae+unet+whisper+face-parser ready in {time.time() - _t_unet:.1f}s on {DEVICE} "
      f"({WEIGHT_DTYPE})", flush=True)
print(f"[boot] lipsync-musetalk ready, {time.time() - _t_models:.1f}s total", flush=True)

# Fixed v1.5 defaults from upstream's CLI (not part of this handler's input contract):
EXTRA_MARGIN = 10
AUDIO_PAD_LEFT = 2
AUDIO_PAD_RIGHT = 2
BATCH_SIZE = 8
PARSING_MODE = "jaw"


def _fetch_to_file(src: str, dest_path: str, timeout: float = 60.0) -> None:
    """A job input is a data: URL or an http(s) URL; write its bytes to dest_path."""
    if src.startswith("data:"):
        _, b64 = src.split(",", 1)
        data = base64.b64decode(b64)
    elif src.startswith("http://") or src.startswith("https://"):
        with urllib.request.urlopen(src, timeout=timeout) as r:  # noqa: S310 - job input, by design
            data = r.read()
    else:
        raise ValueError("must be a data: URL or an http(s) URL")
    with open(dest_path, "wb") as f:
        f.write(data)


def _run_ffmpeg(args: list) -> None:
    proc = subprocess.run(["ffmpeg", *args], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {proc.stderr[-2000:]}")


@torch.no_grad()
def _lip_sync(video_path: str, audio_path: str, bbox_shift: int, fps: float, work_dir: str):
    """The per-task body of upstream's scripts/inference.py main(), against the models already
    loaded at import time. Same function calls, same order, same v1.5 defaults; the only
    intentional difference from upstream is noted inline."""
    frames_dir = os.path.join(work_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    _run_ffmpeg(["-v", "fatal", "-i", video_path, "-start_number", "0", f"{frames_dir}/%08d.png"])
    input_img_list = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    if not input_img_list:
        raise RuntimeError("ffmpeg extracted no frames from the source video")

    whisper_input_features, librosa_length = AUDIO_PROCESSOR.get_audio_feature(audio_path)
    whisper_chunks = AUDIO_PROCESSOR.get_whisper_chunk(
        whisper_input_features, DEVICE, WEIGHT_DTYPE, WHISPER, librosa_length,
        fps=fps, audio_padding_length_left=AUDIO_PAD_LEFT, audio_padding_length_right=AUDIO_PAD_RIGHT,
    )

    coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
    source_height = frame_list[0].shape[0]

    # Faithful to upstream: coord_list/frame_list keep one entry per source frame, including a
    # coord_placeholder for any frame where no face was found -- only input_latent_list skips
    # those. frame_list_cycle/coord_list_cycle below are built from the FULL (unfiltered) lists,
    # exactly as scripts/inference.py's main() does, even though that means they can end up a
    # different length than input_latent_list_cycle if some but not all frames lack a face (the
    # datagen/padding loops below then index each cycle by its own modulo, so the two cycles can
    # drift out of step on that specific edge case -- a pre-existing upstream quirk, not
    # introduced here; a face-less video throughout is caught by the guard just below instead).
    input_latent_list = []
    for bbox, frame in zip(coord_list, frame_list):
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        y2 = min(y2 + EXTRA_MARGIN, source_height)  # v1.5's extra_margin, same clamp upstream applies
        crop_frame = cv2.resize(frame[y1:y2, x1:x2], (256, 256), interpolation=cv2.INTER_LANCZOS4)
        input_latent_list.append(VAE.get_latents_for_unet(crop_frame))
    if not input_latent_list:
        # Upstream's main() has no equivalent guard here: a face-less video reaches datagen's
        # `% len(vae_encode_latents)` with an empty list and dies on a ZeroDivisionError inside
        # the try/except that just prints and moves to the next yaml task. One job, one clear
        # error, is better than that for a caller waiting on this job.
        raise RuntimeError("no face detected in the source video")

    # Ping-pong cycle: datagen indexes into this with `% len(...)`, so audio longer than the
    # source video keeps bouncing forward-reverse-forward through it; audio shorter than the
    # video never asks for more than len(whisper_chunks) frames, i.e. the video is trimmed to
    # the audio. Both are upstream's own mechanism, unchanged.
    frame_list_cycle = frame_list + frame_list[::-1]
    coord_list_cycle = coord_list + coord_list[::-1]
    input_latent_list_cycle = input_latent_list + input_latent_list[::-1]

    gen = datagen(whisper_chunks=whisper_chunks, vae_encode_latents=input_latent_list_cycle,
                  batch_size=BATCH_SIZE, delay_frame=0, device=DEVICE)
    res_frame_list = []
    for whisper_batch, latent_batch in gen:
        audio_feature_batch = PE(whisper_batch)
        latent_batch = latent_batch.to(dtype=UNET.model.dtype)
        pred_latents = UNET.model(latent_batch, TIMESTEPS, encoder_hidden_states=audio_feature_batch).sample
        res_frame_list.extend(VAE.decode_latents(pred_latents))

    result_img_dir = os.path.join(work_dir, "result")
    os.makedirs(result_img_dir, exist_ok=True)
    for i, res_frame in enumerate(res_frame_list):
        bbox = coord_list_cycle[i % len(coord_list_cycle)]
        ori_frame = copy.deepcopy(frame_list_cycle[i % len(frame_list_cycle)])
        x1, y1, x2, y2 = bbox
        try:
            res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
        except Exception:  # noqa: BLE001 - matches upstream: an unresizable frame is skipped, not fatal
            continue
        combine_frame = get_image(ori_frame, res_frame, [x1, y1, x2, y2], mode=PARSING_MODE, fp=FACE_PARSER)
        cv2.imwrite(f"{result_img_dir}/{i:08d}.png", combine_frame)

    temp_vid_path = os.path.join(work_dir, "video_only.mp4")
    _run_ffmpeg(["-y", "-v", "warning", "-r", str(fps), "-f", "image2", "-i", f"{result_img_dir}/%08d.png",
                 "-vcodec", "libx264", "-vf", "format=yuv420p", "-crf", "18", temp_vid_path])

    # Final mux, per this lane's own contract (h264 yuv420p + aac, +faststart) rather than
    # upstream's bare `ffmpeg -i audio -i video out.mp4` (which leaves the audio codec to
    # ffmpeg's defaults): copy the already-h264 video stream, encode audio to aac, and make the
    # file streamable. -shortest is a safety net -- video length already matches the audio by
    # construction above.
    output_path = os.path.join(work_dir, "output.mp4")
    _run_ffmpeg(["-y", "-v", "warning", "-i", temp_vid_path, "-i", audio_path,
                 "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                 "-movflags", "+faststart", "-shortest", output_path])

    height, width = frame_list[0].shape[:2]
    duration_seconds = len(res_frame_list) / fps
    return output_path, duration_seconds, width, height


def handler(job):
    job_input = job.get("input") or {}
    video_src = job_input.get("video")
    audio_src = job_input.get("audio")
    if not video_src:
        return {"error": "video is required"}
    if not audio_src:
        return {"error": "audio is required"}

    try:
        bbox_shift = int(job_input.get("bbox_shift", 0))
    except (TypeError, ValueError):
        return {"error": f"bbox_shift must be an int, got {job_input.get('bbox_shift')!r}"}
    try:
        max_seconds = min(float(job_input.get("max_seconds", 120)), MAX_SECONDS_HARD_CAP)
    except (TypeError, ValueError):
        return {"error": f"max_seconds must be a number, got {job_input.get('max_seconds')!r}"}

    started = time.time()
    work_dir = None
    try:
        # Everything from here down is wrapped: per the contract, a failure comes back as
        # {"error": ...} and this function never raises, including failures too unlikely to
        # have their own specific except clause below (e.g. /tmp full when mkdtemp runs).
        work_dir = tempfile.mkdtemp(prefix="job_", dir=os.path.join(RUN_DIR, "jobs"))
        video_path = os.path.join(work_dir, "input_video.mp4")
        audio_path = os.path.join(work_dir, "input_audio.wav")
        try:
            _fetch_to_file(video_src, video_path)
        except Exception as e:  # noqa: BLE001
            return {"error": f"could not read video: {type(e).__name__}: {e}"}
        try:
            _fetch_to_file(audio_src, audio_path)
        except Exception as e:  # noqa: BLE001
            return {"error": f"could not read audio: {type(e).__name__}: {e}"}

        # librosa, not soundfile: the rest of this pipeline (AudioProcessor.get_audio_feature)
        # already requires librosa to be able to load this exact file, so this is the real gate,
        # not a weaker stand-in for it -- and it's cheap, so it runs before any GPU work.
        import librosa
        audio_duration = librosa.get_duration(path=audio_path)
        if audio_duration > max_seconds:
            return {"error": f"audio is {audio_duration:.1f}s, exceeds max_seconds={max_seconds:.0f}"}

        if job_input.get("fps") is not None:
            try:
                fps = float(job_input["fps"])
            except (TypeError, ValueError):
                return {"error": f"fps must be a number, got {job_input.get('fps')!r}"}
        else:
            fps = get_video_fps(video_path)
            if not fps or fps <= 0:
                fps = 25.0  # upstream's own fallback for a video it could not read an fps from
        fps = max(1.0, min(fps, 30.0))

        try:
            output_path, duration_seconds, width, height = _lip_sync(
                video_path, audio_path, bbox_shift, fps, work_dir)
        except Exception as e:  # noqa: BLE001 - surface the reason, never die silently
            return {"error": f"{type(e).__name__}: {e}"}

        with open(output_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode()
        return {
            "video_url": "data:video/mp4;base64," + video_b64,
            "duration_seconds": round(duration_seconds, 2),
            "width": width,
            "height": height,
            "fps": fps,
            "generation_seconds": round(time.time() - started, 2),
        }
    except Exception as e:  # noqa: BLE001 - the outermost net; see comment above
        return {"error": f"{type(e).__name__}: {e}"}
    finally:
        if work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)


import runpod  # noqa: E402

runpod.serverless.start({"handler": handler})
