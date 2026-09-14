"""HunyuanVideo-I2V - a second top-tier image-to-video opinion, at Model Caching size.

Why this lane exists: the brief assumed HunyuanVideo needed a 371.8GB selective download.
It does not. That figure belongs to `tencent/HunyuanVideo-1.5`, a different model. The
original I2V weights are 30.2GB native, and this diffusers port is 43.6GB - inside the band
Runpod's Model Caching serves. So this lane needs **no network volume**, which also means it
is **not pinned to EUR-IS-1**, where the A14B lane is stuck behind thin GPU stock.

That makes it the hedge: if EUR-IS-1 has no A100 free, this still runs anywhere.

LICENCE - READ BEFORE PRODUCTION USE. The repo loaded here is a community re-upload that
declares `base_model: tencent/HunyuanVideo-I2V` and carries **no licence tag of its own**.
A missing tag on a re-upload changes nothing: the weights are Tencent's and the **Tencent
Hunyuan Community License** governs. Its Territory is worldwide **excluding the European
Union, the United Kingdom and South Korea** - the United States IS inside the grant, which
is why this is usable where MiniMax H3 is not. But §4(c) binds *Outputs*, not just weights:
generated video may not be displayed outside the Territory. EU/UK visitors to a public
site are the residual exposure. See stack-docs/VIDEO-BENCHMARK.md §6.
"""

import base64
import glob
import io
import os
import subprocess
import tempfile
import time
import urllib.request

# Before any heavy import, so "no logs at all" (died pre-Python) stays distinguishable from
# "printed then stopped" (died during import). Those need opposite fixes.
print("[boot] hunyuan i2v handler starting", flush=True)

import torch

print(f"[boot] torch {torch.__version__} cuda={torch.cuda.is_available()}", flush=True)

MODEL_ID = os.environ.get("MODEL_ID", "hunyuanvideo-community/HunyuanVideo-I2V")
CACHE_ROOT = os.environ.get("HF_CACHE_ROOT", "/runpod-volume/huggingface-cache/hub")
FPS = int(os.environ.get("OUTPUT_FPS", "24"))
OFFLOAD = os.environ.get("OFFLOAD", "model")


def find_snapshot(model_id: str) -> str:
    """Locate the snapshot Runpod's Model Caching mounted for this repo."""
    repo_dir = "models--" + model_id.replace("/", "--")
    snaps = [p for p in glob.glob(os.path.join(CACHE_ROOT, repo_dir, "snapshots", "*"))
             if os.path.isdir(p)]
    if not snaps:
        raise RuntimeError(
            f"No cached snapshot for {model_id} under {CACHE_ROOT}. Point the endpoint's "
            "--model-reference at this repo and redeploy."
        )
    return max(snaps, key=os.path.getmtime)


_t0 = time.time()
_snapshot = find_snapshot(MODEL_ID)
print(f"[boot] {MODEL_ID} -> {_snapshot}", flush=True)

from diffusers import HunyuanVideoImageToVideoPipeline  # noqa: E402
from diffusers.models import HunyuanVideoTransformer3DModel  # noqa: E402

# The transformer is loaded separately in bf16 while the rest of the pipeline stays fp16.
# This is diffusers' documented configuration for HunyuanVideo and it is not cosmetic: the
# 13B transformer overflows in fp16 on long clips and returns black frames, while the VAE
# and text encoders are fine at fp16 and cost less memory there.
transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    _snapshot, subfolder="transformer", torch_dtype=torch.bfloat16,
)
PIPE = HunyuanVideoImageToVideoPipeline.from_pretrained(
    _snapshot, transformer=transformer, torch_dtype=torch.float16, local_files_only=True,
)

# The VAE decode is the memory spike, not the denoise: it reconstructs every frame at full
# resolution at once. Tiling trades a little speed for the difference between running and
# OOMing on a 48GB card at 720p.
try:
    PIPE.vae.enable_tiling()
    print("[boot] vae tiling enabled", flush=True)
except Exception as e:
    print(f"[boot] vae tiling unavailable: {e}", flush=True)

if OFFLOAD == "sequential":
    PIPE.enable_sequential_cpu_offload()
elif OFFLOAD == "model":
    PIPE.enable_model_cpu_offload()
else:
    PIPE.to("cuda")
print(f"[boot] offload={OFFLOAD}", flush=True)

try:
    PIPE.set_progress_bar_config(disable=True)
except Exception:
    pass
print(f"[boot] ready in {time.time() - _t0:.1f}s", flush=True)


def _load_image(job_input):
    src = (job_input.get("image_url") or job_input.get("image_base64")
           or job_input.get("image"))
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


def _fit(image, width: int, height: int):
    """Cover-crop then resize. Letterboxed input makes the model animate the padding."""
    from PIL import Image
    iw, ih = image.size
    scale = max(width / iw, height / ih)
    nw, nh = round(iw * scale), round(ih * scale)
    image = image.resize((nw, nh), Image.LANCZOS)
    left, top = (nw - width) // 2, (nh - height) // 2
    return image.crop((left, top, left + width, top + height))


def _frames_to_mp4(frames, fps: int) -> bytes:
    import numpy as np
    arr = [np.asarray(f.convert("RGB")) for f in frames]
    h, w = arr[0].shape[:2]
    out = os.path.join(tempfile.gettempdir(), f"v{int(time.time() * 1000)}.mp4")
    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps), "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "medium", "-crf", "17",
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
    image = _load_image(job_input)
    if image is None:
        return {"error": "this lane is image-to-video: supply image_url or image_base64"}

    started = time.time()
    try:
        width = int(job_input.get("width", 1280)) // 16 * 16
        height = int(job_input.get("height", 720)) // 16 * 16
        image = _fit(image, width, height)

        # 4n+1 frames: the causal temporal VAE encodes in groups of four plus a keyframe.
        frames = int(job_input.get("num_frames", 121))
        frames = max(5, (frames - 1) // 4 * 4 + 1)

        kwargs = {
            "image": image,
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_frames": frames,
            "num_inference_steps": int(job_input.get("num_inference_steps", 30)),
            # HunyuanVideo is distribution-guided rather than CFG-guided; its reference
            # setting is 6.0 and it does not behave like a Wan/SDXL guidance scale.
            "guidance_scale": float(job_input.get("guidance_scale", 6.0)),
        }
        seed = int(job_input.get("seed", -1))
        if seed >= 0:
            kwargs["generator"] = torch.Generator(device="cpu").manual_seed(seed)

        result = PIPE(**kwargs)
        out_frames = result.frames
        # Same guard as vid-wan-a14b: current diffusers returns `.frames` as a numpy array
        # (batch, frames, H, W, C), and `if not <ndarray>` raises after the generation has
        # already been paid for. This lane still returned lists on 2026-09-14, but it shares
        # the same unbounded diffusers pin, so it is one dependency bump from the same failure.
        import numpy as np
        from PIL import Image
        if isinstance(out_frames, np.ndarray):
            if out_frames.ndim == 5:
                out_frames = out_frames[0]
            if out_frames.dtype != np.uint8:
                out_frames = (np.clip(out_frames, 0.0, 1.0) * 255.0).round().astype(np.uint8)
            out_frames = [Image.fromarray(f) for f in out_frames]
        elif isinstance(out_frames, list) and out_frames and isinstance(out_frames[0], list):
            out_frames = out_frames[0]
        if out_frames is None or len(out_frames) == 0:
            return {"error": "pipeline returned no frames"}

        fps = int(job_input.get("fps", FPS))
        mp4 = _frames_to_mp4(out_frames, fps)
        return {
            "video_url": "data:video/mp4;base64," + base64.b64encode(mp4).decode(),
            "model": MODEL_ID,
            "licence": "Tencent Hunyuan Community (Territory excludes EU/UK/South Korea; "
                       "binds Outputs, not just weights)",
            "dtype": "bf16-transformer/fp16-rest",
            "frames": len(out_frames),
            "fps": fps,
            "width": width,
            "height": height,
            "seconds": round(len(out_frames) / fps, 2),
            "steps": kwargs["num_inference_steps"],
            "seed": seed,
            "generation_seconds": round(time.time() - started, 2),
        }
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        return {"error": f"CUDA OOM: {e}. Set OFFLOAD=sequential or drop to 832x480."}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


import runpod  # noqa: E402

runpod.serverless.start({"handler": handler})
