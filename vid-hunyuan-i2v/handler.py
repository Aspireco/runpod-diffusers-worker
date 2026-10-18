"""HunyuanVideo-I2V - a second top-tier image-to-video opinion, for the Azure GPU fleet lane
`hunyuan-i2v` (A100 80GB PCIe).

Uses the diffusers port `hunyuanvideo-community/HunyuanVideo-I2V` (43.6GB) directly through
`diffusers.HunyuanVideoImageToVideoPipeline` -- there is no vendored model code here, unlike
vid-wan-animate or workers/threed-hunyuan3d: this is exactly the official diffusers recipe for
the model, called through its public pipeline API.

This was originally a RunPod worker that resolved its weights through RunPod's own Model Caching
(a MODEL_ID env var plus a glob-search of a HuggingFace-shaped cache under
/runpod-volume/huggingface-cache/hub/, see git history). That mechanism is RunPod-specific and
does not exist on Azure. Weights now come from a plain flat directory, HUNYUAN_I2V_DIR, staged by
the fetch job from gpu/manifests/hunyuan-i2v.tsv -- the same convention vid-wan-animate and
vid-ltx25 use -- loaded with `local_files_only=True` and no network call of any kind
(HF_HUB_OFFLINE=1 is also set in the Dockerfile as a second line of defence).

LICENCE - READ BEFORE PRODUCTION USE. The repo loaded here is a community re-upload that
declares `base_model: tencent/HunyuanVideo-I2V` and carries **no licence tag of its own**. A
missing tag on a re-upload changes nothing: the weights are Tencent's and the **Tencent Hunyuan
Community License** governs. Its Territory is worldwide **excluding the European Union, the
United Kingdom and South Korea** -- the United States IS inside the grant. S4(c) binds *Outputs*,
not just weights: generated video may not be displayed outside the Territory. See README.md's
licence table (this is why the image lives only in azswmcacr/East US 2 and runs only in the new
Canada Central environment).
"""

import base64
import io
import os
import subprocess
import tempfile
import time

import safe_fetch

# Before any heavy import, so "no logs at all" (died pre-Python) stays distinguishable from
# "printed then stopped" (died during import). Those need opposite fixes.
print("[boot] hunyuan i2v handler starting", flush=True)

import torch  # noqa: E402

print(f"[boot] torch {torch.__version__} cuda={torch.cuda.is_available()}", flush=True)

HUNYUAN_I2V_DIR = os.environ.get("HUNYUAN_I2V_DIR", "/runpod-volume/hunyuan-i2v")
FPS = int(os.environ.get("OUTPUT_FPS", "24"))
OFFLOAD = os.environ.get("OFFLOAD", "model")

# One representative file per component the handler actually loads, not all 28 manifest entries
# (gpu/manifests/hunyuan-i2v.tsv lists the exact set; this is a fast boot check, not a full
# integrity scan -- same pattern as vid-wan-animate's _REQUIRED list).
_REQUIRED = [
    "model_index.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "text_encoder/model.safetensors.index.json",
    "text_encoder_2/model.safetensors",
    "vae/diffusion_pytorch_model.safetensors",
    os.path.join("tokenizer", "tokenizer_config.json"),
    os.path.join("tokenizer_2", "vocab.json"),
]

_t0 = time.time()
for _rel in _REQUIRED:
    _full = os.path.join(HUNYUAN_I2V_DIR, _rel)
    if not os.path.isfile(_full):
        raise RuntimeError(
            f"{_full} is missing: run the fetch job for lane hunyuan-i2v "
            f"(gpu/manifests/hunyuan-i2v.tsv) before starting this worker."
        )
print(f"[boot] weights present at {HUNYUAN_I2V_DIR}", flush=True)

from diffusers import HunyuanVideoImageToVideoPipeline  # noqa: E402
from diffusers.models import HunyuanVideoTransformer3DModel  # noqa: E402

# The transformer is loaded separately in bf16 while the rest of the pipeline stays fp16. This is
# diffusers' documented configuration for HunyuanVideo and it is not cosmetic: the 13B transformer
# overflows in fp16 on long clips and returns black frames, while the VAE and text encoders are
# fine at fp16 and cost less memory there.
transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    HUNYUAN_I2V_DIR, subfolder="transformer", torch_dtype=torch.bfloat16, local_files_only=True,
)
PIPE = HunyuanVideoImageToVideoPipeline.from_pretrained(
    HUNYUAN_I2V_DIR, transformer=transformer, torch_dtype=torch.float16, local_files_only=True,
)

# The VAE decode is the memory spike, not the denoise: it reconstructs every frame at full
# resolution at once. Tiling trades a little speed for the difference between running and OOMing
# at 720p. On this lane's A100 80GB (vs. the original worker's target A6000 48GB) there is more
# headroom either way, but tiling is free in time cost here and was already the proven setting, so
# it stays on rather than being relaxed without a GPU job to prove relaxing it is safe (task scope
# excludes starting one).
try:
    PIPE.vae.enable_tiling()
    print("[boot] vae tiling enabled", flush=True)
except Exception as e:
    print(f"[boot] vae tiling unavailable: {e}", flush=True)

# Model-level CPU offload swaps whole modules (text encoders / transformer / VAE) between CPU and
# GPU at stage boundaries. The 13B transformer in bf16 is ~26GB, plus the VAE and two text
# encoders in fp16 (~5GB total) and activations for a 121-151 frame/720p clip -- comfortably under
# 80GB even fully resident, so OFFLOAD=model is a safety margin rather than a necessity here, kept
# as the default because it is the setting this lane was already reasoned through on (A6000 48GB,
# where it mattered more) and relaxing it has no measured upside without a GPU job to time it.
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
    """data URL / raw base64 / https:// URL -> PIL Image. Fetching a URL goes through
    safe_fetch.fetch_image_bytes, not urllib directly: job inputs can arrive from an LLM agent,
    so a URL must never be able to reach an internal address (cloud metadata, localhost, RFC1918,
    link-local -- see safe_fetch.py's module docstring). Raises ValueError on a refused or
    unreachable source; the caller is responsible for turning that into {"error": ...}."""
    src = (job_input.get("image_url") or job_input.get("image_base64")
           or job_input.get("image"))
    if not src:
        return None
    from PIL import Image
    data = safe_fetch.fetch_image_bytes(src)
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
    try:
        image = _load_image(job_input)
    except ValueError as e:
        return {"error": f"could not load image: {e}"}
    if image is None:
        return {"error": "this lane is image-to-video: supply image, image_base64 (data URL or "
                         "raw base64) or image_url"}

    started = time.time()
    try:
        # bounded: inputs can come from an agent, and a 4K or 20 s ask would only run to the deadline
        width = min(max(int(job_input.get("width", 1280)), 256), 1280) // 16 * 16
        height = min(max(int(job_input.get("height", 720)), 256), 1280) // 16 * 16
        image = _fit(image, width, height)

        # 4n+1 frames: the causal temporal VAE encodes in groups of four plus a keyframe. Default
        # 129 (~5.4s at the model's native 24fps) per the task brief's own figure -- the original
        # RunPod worker (see git history) defaulted to 121; 129 is also a valid 4n+1 value
        # (4*32+1) and is the one actually asked for here.
        frames = int(job_input.get("num_frames", 129))
        frames = max(5, (min(frames, 129) - 1) // 4 * 4 + 1)

        kwargs = {
            "image": image,
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_frames": frames,
            "num_inference_steps": min(max(int(job_input.get("num_inference_steps", 30)), 1), 50),
            # HunyuanVideo is distribution-guided rather than CFG-guided; its reference setting is
            # 6.0 and it does not behave like a Wan/SDXL guidance scale.
            "guidance_scale": float(job_input.get("guidance_scale", 6.0)),
        }
        seed = int(job_input.get("seed", -1))
        if seed < 0:
            seed = int.from_bytes(os.urandom(4), "big")
        kwargs["generator"] = torch.Generator(device="cpu").manual_seed(seed)

        result = PIPE(**kwargs)
        out_frames = result.frames
        # Current diffusers returns `.frames` as a numpy array (batch, frames, H, W, C), and
        # `if not <ndarray>` raises after the generation has already been paid for -- same guard
        # as vid-wan-a14b.
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
            "width": width,
            "height": height,
            "fps": fps,
            "frames": len(out_frames),
            "seconds": round(len(out_frames) / fps, 2),
            "seed": seed,
            "generation_seconds": round(time.time() - started, 2),
            "model": "hunyuanvideo-community/HunyuanVideo-I2V",
            "licence": "Tencent Hunyuan Community (Territory excludes EU/UK/South Korea; "
                       "binds Outputs, not just weights)",
            "dtype": "bf16-transformer/fp16-rest",
            "steps": kwargs["num_inference_steps"],
        }
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        return {"error": f"CUDA OOM: {e}. Set OFFLOAD=sequential or drop to 832x480."}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


import runpod  # noqa: E402

runpod.serverless.start({"handler": handler})
