"""Wan 2.2 I2V-A14B - the full-precision 14B-active MoE, the tier above the deployed 5B.

Why this model: the stack currently runs Wan2.2-TI2V-5B, which is the small member of its
own family, and the quality complaint is about exactly that. The A14B is the same family's
flagship - a mixture-of-experts with two 14B-active transformers, one trained for
high-noise timesteps and one for low - under a plain apache-2.0 licence with no territory
carve-out, no revenue cap and no attribution clause. It is the only top-tier open video
model with a licence that clean (see stack-docs/VIDEO-BENCHMARK.md for the comparison).

Weights come from the network volume, not Model Caching: the repo is 126.2GB and Model
Caching is sized for the 35-60GB band. That pins this endpoint to EUR-IS-1, which is a real
constraint - that DC listed only two GPU types in stock when this was built.

The 126.2GB figure is a storage artefact, not a quality one. Both experts ship as fp32
checkpoints (57.1GB each for 14B parameters = 4 bytes/param) while the model was trained
and is served in bf16. Loading at bf16 is the normal, lossless-for-inference path and
halves resident VRAM to ~28.6GB per expert. Do not confuse that cast with quantization.
"""

import base64
import glob
import inspect
import io
import os
import subprocess
import tempfile
import time
import urllib.request

# Printed before any heavy import so "no logs at all" (died pre-Python, almost always a
# CUDA/driver mismatch) stays distinguishable from "printed then stopped" (died during
# import). Those two have opposite fixes and look identical in Runpod's console otherwise.
print("[boot] wan a14b handler starting", flush=True)

import torch

print(f"[boot] torch {torch.__version__} cuda={torch.cuda.is_available()}", flush=True)

MODEL_DIR_ENV = os.environ.get("MODEL_DIR", "")
MODEL_ID = os.environ.get("MODEL_ID", "Wan-AI/Wan2.2-I2V-A14B-Diffusers")
FPS = int(os.environ.get("OUTPUT_FPS", "16"))          # Wan 2.2 is a 16fps model
OFFLOAD = os.environ.get("OFFLOAD", "model")           # model | sequential | none
DTYPE = os.environ.get("DTYPE", "bf16")                # bf16 | fp16 | fp8

# The volume mounts at /runpod-volume on a serverless worker but /workspace on a pod, and
# the loader script writes from a pod. Both spellings are checked so the same image works
# either way, with the Model Caching layout kept as a last resort.
CANDIDATES = [
    MODEL_DIR_ENV,
    "/runpod-volume/wan22-i2v-a14b",
    "/workspace/wan22-i2v-a14b",
]


def resolve_model_dir() -> str:
    for c in CANDIDATES:
        if c and os.path.isdir(os.path.join(c, "transformer")):
            return c
    cache_root = os.environ.get("HF_CACHE_ROOT", "/runpod-volume/huggingface-cache/hub")
    repo_dir = "models--" + MODEL_ID.replace("/", "--")
    snaps = [p for p in glob.glob(os.path.join(cache_root, repo_dir, "snapshots", "*"))
             if os.path.isdir(p)]
    if snaps:
        return max(snaps, key=os.path.getmtime)
    raise RuntimeError(
        "Wan A14B weights not found. Looked for a 'transformer' directory under "
        + ", ".join(p for p in CANDIDATES if p)
        + f" and for a Model Caching snapshot of {MODEL_ID}. Attach network volume "
        "yweuz29h2k (EUR-IS-1) and run vid-wan-a14b/load_weights.sh first."
    )


_t0 = time.time()
MODEL_DIR = resolve_model_dir()
print(f"[boot] weights at {MODEL_DIR}", flush=True)

# Fail loudly and early if the download dropped an expert. A missing transformer_2 leaves a
# pipeline that loads fine and then produces mush for every timestep below the MoE
# boundary, which is the hardest possible failure to diagnose from the output alone.
for sub in ("transformer", "transformer_2", "text_encoder", "vae"):
    if not os.path.isdir(os.path.join(MODEL_DIR, sub)):
        raise RuntimeError(f"{MODEL_DIR} is missing '{sub}' - the weight download is incomplete")

from diffusers import AutoencoderKLWan, WanImageToVideoPipeline  # noqa: E402

# A typo in the DTYPE env var must not become a KeyError at module import -- that is a
# crash-loop with no useful log, which is the exact failure class this stack keeps paying
# for. Fall back loudly instead.
_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp8": torch.bfloat16}
if DTYPE not in _DTYPES:
    print(f"[boot] unknown DTYPE={DTYPE!r}, falling back to bf16 "
          f"(valid: {sorted(_DTYPES)})", flush=True)
    DTYPE = "bf16"
_TORCH_DTYPE = _DTYPES[DTYPE]

# The VAE stays fp32 on purpose. It is 0.5GB - nothing against a 57GB transformer - and
# Wan's temporal VAE is where half precision shows up first, as colour drift and blocking
# in flat areas. This is diffusers' documented configuration for the Wan family.
vae = AutoencoderKLWan.from_pretrained(MODEL_DIR, subfolder="vae",
                                       torch_dtype=torch.float32, local_files_only=True)

PIPE = WanImageToVideoPipeline.from_pretrained(
    MODEL_DIR, vae=vae, torch_dtype=_TORCH_DTYPE, local_files_only=True,
)

# What actually ran, as opposed to what was asked for. If torchao is missing the pipeline
# silently stays bf16, and reporting the *requested* dtype would label a bf16 run "fp8" in
# the results table -- corrupting the one comparison this lane exists to make.
EFFECTIVE_DTYPE = DTYPE

if DTYPE == "fp8":
    # The quantized arm of the benchmark. torchao casts the two transformers' linear layers
    # to fp8 in place, roughly halving their resident footprint again (~14GB per expert) so
    # both fit a 48GB card without offload thrash. The quality delta is the thing being
    # measured, so this stays a runtime switch rather than a separate image.
    try:
        from torchao.quantization import float8_weight_only, quantize_
        for name in ("transformer", "transformer_2"):
            mod = getattr(PIPE, name, None)
            if mod is not None:
                quantize_(mod, float8_weight_only())
        print("[boot] fp8 weight-only quantization applied to both experts", flush=True)
    except Exception as e:
        EFFECTIVE_DTYPE = "bf16 (fp8 requested, unavailable)"
        print(f"[boot] fp8 requested but unavailable ({type(e).__name__}: {e}); staying bf16",
              flush=True)

# Two 28.6GB experts plus an 11GB text encoder is ~68GB resident, so anything short of an
# 80GB card needs offload. model-level offload keeps whole submodules on the GPU and swaps
# them at boundaries (fast, wants ~30GB); sequential goes layer by layer (slow, ~10GB) and
# is the only thing that fits a 24GB card.
if OFFLOAD == "sequential":
    PIPE.enable_sequential_cpu_offload()
    print("[boot] sequential cpu offload", flush=True)
elif OFFLOAD == "model":
    PIPE.enable_model_cpu_offload()
    print("[boot] model cpu offload", flush=True)
else:
    PIPE.to("cuda")
    print("[boot] fully resident on gpu", flush=True)

try:
    PIPE.set_progress_bar_config(disable=True)
except Exception:
    pass

_SIG = set(inspect.signature(PIPE.__call__).parameters)
print(f"[boot] ready in {time.time() - _t0:.1f}s | dtype={EFFECTIVE_DTYPE} offload={OFFLOAD} "
      f"dual_guidance={'guidance_scale_2' in _SIG}", flush=True)


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
    """Cover-crop to the target aspect, then resize. Wan is sensitive to letterboxing - a
    padded input produces a video that animates the padding as if it were scene content."""
    from PIL import Image
    iw, ih = image.size
    scale = max(width / iw, height / ih)
    nw, nh = round(iw * scale), round(ih * scale)
    image = image.resize((nw, nh), Image.LANCZOS)
    left, top = (nw - width) // 2, (nh - height) // 2
    return image.crop((left, top, left + width, top + height))


def _frames_to_mp4(frames, fps: int) -> bytes:
    """Encode PIL frames with ffmpeg over a pipe. A 720p 81-frame clip as a PNG sequence is
    several hundred MB on a 20GB container disk; piping keeps nothing but the mp4."""
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
        return {"error": "this lane is image-to-video: supply image_url or image_base64. "
                         "For text-to-video use the T2V-A14B lane."}

    started = time.time()
    try:
        # 16 is the VAE spatial compression times the transformer patch size; anything not
        # divisible by it is silently rounded by the pipeline and the returned clip no
        # longer matches the aspect ratio that was asked for.
        width = int(job_input.get("width", 1280)) // 16 * 16
        height = int(job_input.get("height", 720)) // 16 * 16
        image = _fit(image, width, height)

        # 4n+1 frames: the temporal VAE encodes in groups of four plus a keyframe, so 81
        # frames is 5.06s at 16fps and 80 would be silently truncated.
        frames = int(job_input.get("num_frames", 81))
        frames = max(5, (frames - 1) // 4 * 4 + 1)

        kwargs = {
            "image": image,
            "prompt": prompt,
            "negative_prompt": job_input.get(
                "negative_prompt",
                "static, still image, low quality, blurry, watermark, text, distorted, "
                "deformed hands, jpeg artifacts, oversaturated",
            ),
            "width": width,
            "height": height,
            "num_frames": frames,
            "num_inference_steps": int(job_input.get("num_inference_steps", 40)),
            "guidance_scale": float(job_input.get("guidance_scale", 3.5)),
        }

        # The MoE takes two guidance values, one per expert. Older diffusers exposes only
        # the single scale, so this is introspected rather than assumed - passing an
        # unknown kwarg to a diffusers pipeline is a hard TypeError, not a warning.
        if "guidance_scale_2" in _SIG:
            kwargs["guidance_scale_2"] = float(job_input.get("guidance_scale_2", 3.5))

        # 720p wants a higher flow shift than the 3.0 baked into the scheduler config;
        # Wan's own reference settings use 5.0 at this resolution. Left adjustable because
        # it is the single most effective knob for "too much motion" vs "barely moves".
        shift = job_input.get("flow_shift")
        if shift is None:
            shift = 5.0 if height >= 700 else 3.0
        # A diffusers scheduler config is a FrozenDict, so assigning to it raises and the
        # setting would silently never apply -- the clip would come back subtly wrong with
        # only a warning in the log. Copy to a plain dict and rebuild the scheduler instead.
        try:
            cfg = dict(PIPE.scheduler.config)
            if cfg.get("flow_shift") != float(shift):
                cfg["flow_shift"] = float(shift)
                PIPE.scheduler = PIPE.scheduler.__class__.from_config(cfg)
        except Exception as e:
            print(f"[warn] flow_shift {shift} not applied: {type(e).__name__}: {e}",
                  flush=True)

        seed = int(job_input.get("seed", -1))
        if seed >= 0:
            kwargs["generator"] = torch.Generator(device="cpu").manual_seed(seed)

        result = PIPE(**kwargs)
        out_frames = result.frames
        # diffusers changed what `.frames` is. Older releases returned a list of lists of
        # PIL images; current ones return a numpy array of shape (batch, frames, H, W, C),
        # float in [0, 1]. The old `if not out_frames:` guard raises "The truth value of an
        # array with more than one element is ambiguous" on the new type -- AFTER the full
        # generation has run and been paid for. Found live 2026-09-14 on :latest, which the
        # unbounded `diffusers>=0.31` pin let drift. Normalise both shapes to a list of PIL
        # frames, which is what _frames_to_mp4 expects.
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
            "dtype": EFFECTIVE_DTYPE,
            "offload": OFFLOAD,
            "frames": len(out_frames),
            "fps": fps,
            "width": width,
            "height": height,
            "seconds": round(len(out_frames) / fps, 2),
            "steps": kwargs["num_inference_steps"],
            "flow_shift": float(shift),
            "seed": seed,
            "generation_seconds": round(time.time() - started, 2),
        }
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        return {"error": f"CUDA OOM: {e}. Set OFFLOAD=sequential, drop to 832x480, "
                         "or move the endpoint to an 80GB card."}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


import runpod  # noqa: E402

runpod.serverless.start({"handler": handler})
