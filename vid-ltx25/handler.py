"""LTX-2.5 (Lightricks): text or image to video WITH sound, for the Azure GPU fleet lane `ltx25` (A100 80GB).

The diffusers 0.41 reference recipe for Lightricks/LTX-2.5-Diffusers, distilled and two-stage: stage 1 at half
size with the 8 distilled sigmas, the latent upsampler doubles it, stage 2 refines at full size with 3 sigmas,
unguided (guidance 1.0). Video and audio come out of the same pass, and the mp4 carries both.

Licence: the LTX-2 Community Licence (every LTX-2 version since 2026-01-05) is free for any purpose below
US$10M annual revenue counted across all affiliated businesses; John confirmed he clears it (marketing os
stack-docs DECISIONS 0.2). The text encoder is Gemma 4, under Google's Gemma terms.

Weights are read from the Azure Files share (MODEL_DIR), staged by the fetch job; the image holds none.

Input:  prompt, optional image_base64 | image_url (the first frame), width and height (multiples of 64),
        num_frames (8n+1), frame_rate (24), seed.
Output: video_url (data: mp4 with an AAC track), frames, fps, seconds, width, height, audio, seed.
"""

import base64
import io
import os
import tempfile
import time
import urllib.request

# Printed before any heavy import, so "no logs at all" (died before Python) stays distinguishable from
# "printed then stopped" (died during import).
print("[boot] ltx25 handler starting", flush=True)

import torch  # noqa: E402

print(f"[boot] torch {torch.__version__} cuda={torch.cuda.is_available()}", flush=True)

MODEL_DIR = os.environ.get("MODEL_DIR", "/runpod-volume/ltx25")
OFFLOAD = os.environ.get("OFFLOAD", "model")      # model | sequential | none
FPS = float(os.environ.get("OUTPUT_FPS", "24"))
# Components the distilled two-stage recipe never calls; not loading them saves reading them on every cold start.
SKIP = [c for c in os.environ.get("SKIP_COMPONENTS", "prompt_enhancer,processor").split(",") if c]

for sub in ("transformer", "text_encoder", "connectors", "vae", "audio_vae", "vocoder", "latent_upsampler"):
    if not os.path.isdir(os.path.join(MODEL_DIR, sub)):
        raise RuntimeError(f"{MODEL_DIR} is missing '{sub}': run the fetch job for lane ltx25 first")

_t0 = time.time()
from diffusers import LTX2ConditionPipeline  # noqa: E402
from diffusers.pipelines.ltx2 import LTX2LatentUpsamplePipeline  # noqa: E402
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel  # noqa: E402
from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition  # noqa: E402
from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES  # noqa: E402
from diffusers.utils import encode_video  # noqa: E402
from PIL import Image  # noqa: E402

# One pipeline for text-to-video (no conditions), image-to-video (a first-frame condition) and stage 2. Two
# pipelines sharing modules would each install their own offload hooks on the same weights.
PIPE = LTX2ConditionPipeline.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16, local_files_only=True,
                                             **{c: None for c in SKIP})
UPSAMPLER = LTX2LatentUpsamplerModel.from_pretrained(MODEL_DIR, subfolder="latent_upsampler",
                                                     torch_dtype=torch.bfloat16, local_files_only=True)
UPSAMPLE = LTX2LatentUpsamplePipeline(vae=PIPE.vae, latent_upsampler=UPSAMPLER)

# A 22B distilled DiT plus a Gemma 4 text encoder do not sit on an 80GB card together with activations:
# model-level offload keeps whole modules on the GPU and swaps them at boundaries.
if OFFLOAD == "sequential":
    PIPE.enable_sequential_cpu_offload()
elif OFFLOAD == "model":
    PIPE.enable_model_cpu_offload()
else:
    PIPE.to("cuda")
UPSAMPLE.enable_model_cpu_offload()
PIPE.vae.enable_tiling()
for p in (PIPE, UPSAMPLE):
    try:
        p.set_progress_bar_config(disable=True)
    except Exception:
        pass
AUDIO_RATE = PIPE.vocoder.config.output_sampling_rate
print(f"[boot] ready in {time.time() - _t0:.1f}s | offload={OFFLOAD} skipped={SKIP} audio={AUDIO_RATE} Hz", flush=True)


def _load_image(job_input):
    src = job_input.get("image_url") or job_input.get("image_base64") or job_input.get("image")
    if not src:
        return None
    if src.startswith("http"):
        with urllib.request.urlopen(src, timeout=90) as r:
            data = r.read()
    else:
        raw = src.split(",", 1)[1] if src.startswith("data:") else src
        data = base64.b64decode(raw, validate=True)
    return Image.open(io.BytesIO(data)).convert("RGB")


def _fit(image, width, height):
    """Cover-crop to the target shape, then resize: letterboxing would be animated as part of the scene."""
    w, h = image.size
    scale = max(width / w, height / h)
    image = image.resize((max(width, round(w * scale)), max(height, round(h * scale))), Image.LANCZOS)
    left, top = (image.width - width) // 2, (image.height - height) // 2
    return image.crop((left, top, left + width, top + height))


def handler(job):
    job_input = job.get("input") or {}
    prompt = (job_input.get("prompt") or "").strip()
    if not prompt:
        return {"error": "prompt is required"}
    # Two-stage needs multiples of 64: stage 1 halves each side and must still land on the VAE's 32-pixel grid.
    width = int(job_input.get("width", 1280)) // 64 * 64
    height = int(job_input.get("height", 704)) // 64 * 64
    if width < 256 or height < 256:
        return {"error": f"{width}x{height} is too small; use at least 256 on each side (multiples of 64)"}
    # 8n+1 frames: the VAE encodes time in groups of eight plus a keyframe.
    frames = int(job_input.get("num_frames", 121))
    frames = max(9, (frames - 1) // 8 * 8 + 1)
    rate = float(job_input.get("frame_rate", FPS))
    seed = int(job_input.get("seed", -1))
    if seed < 0:
        seed = int.from_bytes(os.urandom(4), "big")
    generator = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)

    image = _load_image(job_input)
    started = time.time()
    try:
        common = {"prompt": prompt, "num_frames": frames, "frame_rate": rate, "guidance_scale": 1.0,
                  "audio_guidance_scale": 1.0, "generator": generator, "return_dict": False}
        conditions = None
        if image is not None:
            conditions = [LTX2VideoCondition(frames=_fit(image, width // 2, height // 2), index=0, strength=1.0)]
        video_latent, audio_latent = PIPE(conditions=conditions, width=width // 2, height=height // 2,
                                          sigmas=DISTILLED_SIGMA_VALUES, output_type="latent", **common)
        # output_type="latent" already applied the latent statistics; the upsampler wants them denormalized
        upscaled = UPSAMPLE(latents=video_latent, latents_normalized=False, output_type="latent", return_dict=False)[0]
        video, audio = PIPE(latents=upscaled, audio_latents=audio_latent, width=width, height=height,
                            sigmas=STAGE_2_DISTILLED_SIGMA_VALUES, noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
                            output_type="np", **common)
        fd, path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        try:
            encode_video(video[0], fps=rate, audio=audio[0].float().cpu(), audio_sample_rate=AUDIO_RATE,
                         output_path=path)
            data = open(path, "rb").read()
        finally:
            os.unlink(path)
        n = len(video[0])
        return {
            "video_url": "data:video/mp4;base64," + base64.b64encode(data).decode(),
            "model": "Lightricks/LTX-2.5-Diffusers (distilled, two-stage)",
            "mode": "image-to-video" if image is not None else "text-to-video",
            "frames": n, "fps": rate, "seconds": round(n / rate, 2), "width": width, "height": height,
            "audio": True, "seed": seed, "generation_seconds": round(time.time() - started, 2),
        }
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        return {"error": f"CUDA OOM: {e}. Try fewer frames or a smaller size, or OFFLOAD=sequential."}
    except Exception as e:  # surface the reason rather than dying silently
        return {"error": f"{type(e).__name__}: {e}"}


import runpod  # noqa: E402

runpod.serverless.start({"handler": handler})
