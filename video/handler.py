"""Video worker: NVIDIA Cosmos 3 (OpenMDW-1.1) image-to-video and text-to-video.

Chosen over the Wan 2.2 TI2V-5B we had because that is the small model in the family and it
shows. Cosmos 3's distilled image-to-video currently tops the open-weight arena, the repo is
35GB so Runpod's Model Caching can hold it without a network volume, and OpenMDW-1.1 is a
permissive Linux Foundation licence that states commercial use plainly — no territory carve-
out like Hunyuan, no revenue cap like Stability, no gate like LTX-2.5.

Weights come from the host-side cache, not the image: MODEL_ID and the endpoint's
--model-reference must name the same repo.
"""

import base64
import glob
import io
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

# Before any heavy import, so "no logs at all" (died pre-Python) stays distinguishable from
# "printed then stopped" (died during import). Those need opposite fixes.
print("[boot] video handler starting", flush=True)

import torch
print(f"[boot] torch {torch.__version__} cuda={torch.cuda.is_available()}", flush=True)

MODEL_ID = os.environ.get("MODEL_ID", "nvidia/Cosmos3-Nano")
CACHE_ROOT = os.environ.get("HF_CACHE_ROOT", "/runpod-volume/huggingface-cache/hub")
FPS = int(os.environ.get("OUTPUT_FPS", "24"))


def find_snapshot(model_id: str) -> str:
    """Locate the snapshot Runpod's Model Caching mounted for this repo."""
    repo_dir = "models--" + model_id.replace("/", "--")
    snaps = [p for p in glob.glob(os.path.join(CACHE_ROOT, repo_dir, "snapshots", "*"))
             if os.path.isdir(p)]
    if not snaps:
        raise RuntimeError(
            f"No cached snapshot for {model_id} under {CACHE_ROOT}. "
            "Point the endpoint's model reference at this repo and redeploy."
        )
    return max(snaps, key=os.path.getmtime)


_t0 = time.time()
_snapshot = find_snapshot(MODEL_ID)
print(f"[boot] {MODEL_ID} -> {_snapshot}", flush=True)

import diffusers  # noqa: E402
from diffusers import DiffusionPipeline  # noqa: E402

# DO NOT replace this with a bare DiffusionPipeline.from_pretrained call.
# That helper reads "_class_name" out of the repo's model_index.json and getattr()s it off
# the diffusers module. Cosmos3-Nano declares "Cosmos3OmniDiffusersPipeline", and NO diffusers
# version has ever exported a class by that name -- the real one is "Cosmos3OmniPipeline",
# added in v0.39.0. So the generic loader dies with
#   AttributeError: module diffusers has no attribute Cosmos3OmniDiffusersPipeline
# on a correctly downloaded, perfectly good checkpoint. The repo's metadata is simply wrong.
#
# Resolving the class ourselves also keeps MODEL_ID swappable: anything whose declared class
# does exist still loads through the generic path below.
_declared = "?"
try:
    import json as _json
    with open(os.path.join(_snapshot, "model_index.json")) as _f:
        _declared = _json.load(_f).get("_class_name", "?")
except Exception:
    pass

# Known-wrong names in published metadata -> the class diffusers actually ships.
_CLASS_FIXUPS = {"Cosmos3OmniDiffusersPipeline": "Cosmos3OmniPipeline"}
_wanted = _CLASS_FIXUPS.get(_declared, _declared)

_cls = getattr(diffusers, _wanted, None)
if _cls is None:
    _cls = DiffusionPipeline  # generic path; will raise with a clear message if it cannot cope
    print(f"[boot] no class {_wanted!r} in diffusers {diffusers.__version__}, "
          f"falling back to DiffusionPipeline", flush=True)
else:
    print(f"[boot] model declares {_declared!r} -> loading {_wanted!r} "
          f"(diffusers {diffusers.__version__})", flush=True)

# enable_safety_checker=False is REQUIRED here, not a preference. Cosmos3OmniPipeline
# constructs CosmosSafetyChecker unless told not to, and that class lives in the
# cosmos_guardrail package -- which cannot be installed alongside this image at all
# (0.3.x wants transformers>=5, 0.1.0 wants torch==2.6.0; the base has transformers<5 and
# torch 2.8, so pip returns ResolutionImpossible). Passing False skips the import.
# The kwarg only exists on the Cosmos pipelines, so it is applied conditionally to keep
# MODEL_ID swappable.
import inspect as _inspect  # noqa: E402
_kw = {"torch_dtype": torch.bfloat16, "local_files_only": True}
try:
    if "enable_safety_checker" in _inspect.signature(_cls.__init__).parameters:
        _kw["enable_safety_checker"] = False
        print("[boot] safety checker disabled (cosmos_guardrail is uninstallable here)",
              flush=True)
except (TypeError, ValueError):
    pass

PIPE = _cls.from_pretrained(_snapshot, **_kw)
# Video models are far larger than image ones relative to the card; offloading streams
# submodules instead of holding the whole graph resident, which is the difference between
# running on a 48GB card and OOMing during weight load.
if os.environ.get("CPU_OFFLOAD", "1") == "1":
    PIPE.enable_model_cpu_offload()
    print("[boot] cpu offload enabled", flush=True)
else:
    PIPE.to("cuda")
try:
    PIPE.set_progress_bar_config(disable=True)
except Exception:
    pass
print(f"[boot] ready in {time.time() - _t0:.1f}s", flush=True)


def _load_image(job_input):
    src = job_input.get("image_url") or job_input.get("image_base64") or job_input.get("image")
    if not src:
        return None
    from PIL import Image
    if src.startswith("http"):
        with urllib.request.urlopen(src, timeout=90) as r:
            data = r.read()
    else:
        raw = src.split(",", 1)[1] if src.startswith("data:") else src
        data = base64.b64decode(raw, validate=True)
    return Image.open(io.BytesIO(data)).convert("RGB")


def _sound_to_wav(sound, sample_rate: int = 16000):
    """Write a Cosmos `sound` tensor ([C, N]) to a temp WAV, or return None.

    Cosmos 3 is an OMNI model: it generates a soundtrack alongside the video. Dropping it
    would throw away half of what the model produced, and it is the reason this lane can
    answer "video with audio" without MiniMax's territory restrictions."""
    try:
        import numpy as np
        import wave
        a = sound.detach().float().cpu().numpy() if hasattr(sound, "detach") else np.asarray(sound)
        if a.ndim == 1:
            a = a[None, :]
        ch, _ = a.shape
        a = np.clip(a, -1.0, 1.0)
        pcm = (a * 32767.0).astype("<i2").T.tobytes()   # interleave to [N, C]
        path = os.path.join(tempfile.gettempdir(), f"a{int(time.time()*1000)}.wav")
        with wave.open(path, "wb") as w:
            w.setnchannels(ch)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(pcm)
        return path
    except Exception as e:
        print(f"[warn] could not encode sound track: {type(e).__name__}: {e}", flush=True)
        return None


def _frames_to_mp4(frames, fps: int, wav_path=None) -> bytes:
    """Encode PIL frames with ffmpeg, muxing the generated soundtrack when there is one.
    Frames stream in over a pipe rather than landing as a PNG sequence — a few seconds of HD
    would otherwise fill the container disk."""
    import numpy as np
    arr = [np.asarray(f.convert("RGB")) for f in frames]
    h, w = arr[0].shape[:2]
    out = os.path.join(tempfile.gettempdir(), f"v{int(time.time()*1000)}.mp4")
    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps), "-i", "pipe:0",
    ]
    if wav_path:
        # -shortest so a soundtrack longer than the video does not pad the tail with a
        # frozen final frame.
        cmd += ["-i", wav_path, "-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd += [
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", out,
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                         stderr=subprocess.PIPE)
    for a in arr:
        p.stdin.write(a.tobytes())
    p.stdin.close()
    err = p.stderr.read().decode(errors="ignore")[-800:]
    if p.wait() != 0:
        raise RuntimeError(f"ffmpeg failed: {err}")
    data = open(out, "rb").read()
    os.unlink(out)
    return data


def handler(job):
    job_input = job.get("input") or {}
    prompt = (job_input.get("prompt") or "").strip()
    if not prompt:
        return {"error": "prompt is required"}

    started = time.time()
    try:
        kwargs = {
            "prompt": prompt,
            "num_inference_steps": int(job_input.get("num_inference_steps", 35)),
            "guidance_scale": float(job_input.get("guidance_scale", 4.0)),
        }
        neg = (job_input.get("negative_prompt") or "").strip()
        if neg:
            kwargs["negative_prompt"] = neg

        frames = int(job_input.get("num_frames", 121))
        kwargs["num_frames"] = frames

        image = _load_image(job_input)
        if image is not None:
            kwargs["image"] = image           # image-to-video
        else:
            kwargs["width"] = int(job_input.get("width", 1280))
            kwargs["height"] = int(job_input.get("height", 704))

        seed = int(job_input.get("seed", -1))
        if seed >= 0:
            kwargs["generator"] = torch.Generator(device="cuda").manual_seed(seed)

        result = PIPE(**kwargs)
        # Cosmos3OmniPipelineOutput exposes `video`, NOT `frames` -- the older
        # CosmosPipelineOutput used `frames`, and reading only that returned "no frames" on a
        # perfectly good 242-second generation. Check every known field before giving up, and
        # say which fields DID exist if none match, so the next surprise is one log line.
        out_frames = None
        for _attr in ("video", "frames", "videos", "images"):
            _v = getattr(result, _attr, None)
            if _v is not None:
                out_frames = _v
                break
        if out_frames is None:
            _fields = [k for k in dir(result) if not k.startswith("_")]
            return {"error": "pipeline returned no video; output fields were "
                             + ", ".join(_fields[:20])}
        if isinstance(out_frames, list) and out_frames and isinstance(out_frames[0], list):
            out_frames = out_frames[0]

        fps = int(job_input.get("fps", FPS))
        _wav = None
        _sound = getattr(result, "sound", None)
        if _sound is not None:
            _wav = _sound_to_wav(_sound)
            if _wav:
                print("[gen] muxing the model's generated soundtrack", flush=True)
        mp4 = _frames_to_mp4(out_frames, fps, wav_path=_wav)
        if _wav:
            try:
                os.unlink(_wav)
            except OSError:
                pass
        return {
            "video_url": "data:video/mp4;base64," + base64.b64encode(mp4).decode(),
            "model": MODEL_ID,
            "frames": len(out_frames),
            "fps": fps,
            "seconds": round(len(out_frames) / fps, 2),
            "seed": seed,
            "generation_seconds": round(time.time() - started, 2),
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


import runpod  # noqa: E402

runpod.serverless.start({"handler": handler})
