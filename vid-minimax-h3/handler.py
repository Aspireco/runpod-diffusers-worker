"""MiniMax-H3 (MiniMaxAI/MiniMax-H3): text/first-frame-and-last-frame to video WITH audio, for the Azure
GPU fleet lane `minimax-h3` (A100 80GB, 220GiB host RAM on the replica). One transformer denoises video and
audio latents together in a single packed sequence -- there is no separate vocoder and no audio post-pass.

Only the `fl2va` workflow is loaded (text-only `t2va` is a subset of the same transformer partition; the
`ref2va` omni-reference workflow needs the separate `transformer_ref/` partition, which this lane does not
stage -- see README.md's Premade search and gpu/manifests/minimax-h3.tsv). diffusers exposes MiniMax-H3 as
Modular Diffusers blocks only (`ModularPipeline`), there is no classic `DiffusionPipeline` half -- see
https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_h3.

LICENCE -- READ README.md. The MiniMax H3 Community License's Territory excludes the EU, the UK, South
Korea and the UNITED STATES (the US is OUTSIDE the grant here, unlike the Tencent Hunyuan licence this
repo's other video lanes carry -- the US is inside Hunyuan's Territory but outside this one). This is why
this image is built and pushed only to `azswgpuacrca` (Canada Central) and MUST NEVER be built on
`azswmcacr` (East US 2): a build or push to a US registry puts the Works outside the Territory before a
single job has even run. The lane itself is wired to run only in the Canada Central GPU environment.

Offline loading trap (see build_check.py's `offline` check for the proof): `modular_model_index.json`
names every component with `pretrained_model_name_or_path: "MiniMaxAI/MiniMax-H3"` plus a subfolder. Calling
`ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-H3")` would therefore try the Hub for every component
regardless of HF_HUB_OFFLINE. The fix is not a hand-rewrite of that json: diffusers' own
`ModularPipeline.from_pretrained` already rewrites a component's spec to whatever path it was called with,
the instant that path's own `<path>/<subfolder>` already contains that component's files
(`_is_local_component` in diffusers/modular_pipelines/modular_pipeline.py). So calling
`ModularPipeline.from_pretrained(MINIMAX_H3_DIR, ...)` -- the local flat directory itself, not the repo id
-- is enough: every component spec gets its `pretrained_model_name_or_path` overwritten to MINIMAX_H3_DIR
before `load_components()` ever reaches the network path. build_check.py proves this at build time, on a
small decoy directory, without downloading the 144GB of real weights.
"""

import base64
import io
import os
import tempfile
import time

import safe_fetch

# Before any heavy import, so "no logs at all" (died pre-Python) stays distinguishable from
# "printed then stopped" (died during import). Those need opposite fixes.
print("[boot] minimax-h3 handler starting", flush=True)

import torch  # noqa: E402

print(f"[boot] torch {torch.__version__} cuda={torch.cuda.is_available()}", flush=True)

MINIMAX_H3_DIR = os.environ.get("MINIMAX_H3_DIR", "/runpod-volume/minimax-h3")
FPS = int(os.environ.get("OUTPUT_FPS", "24"))
MEMORY_RESERVE_MARGIN = os.environ.get("MEMORY_RESERVE_MARGIN", "12GB")

# One representative file per component the fl2va workflow actually loads, not every entry in
# gpu/manifests/minimax-h3.tsv -- a fast boot check, not a full integrity scan (same pattern as
# vid-hunyuan-i2v's _REQUIRED list). transformer_ref/ is deliberately absent from this list and from the
# manifest: this lane only stages and loads the fl2va/t2va transformer partition.
_REQUIRED = [
    "model_index.json",
    "modular_model_index.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors.index.json",
    "audio_vae/config.json",
    "audio_vae/diffusion_pytorch_model.safetensors",
    "scheduler/scheduler_config.json",
    "audio_scheduler/scheduler_config.json",
    "tokenizer/tokenizer_config.json",
    "processor/preprocessor_config.json",
]

_t0 = time.time()
for _rel in _REQUIRED:
    _full = os.path.join(MINIMAX_H3_DIR, _rel)
    if not os.path.isfile(_full):
        raise RuntimeError(
            f"{_full} is missing: run the fetch job for lane minimax-h3 "
            f"(gpu/manifests/minimax-h3.tsv) before starting this worker."
        )
print(f"[boot] weights present at {MINIMAX_H3_DIR}", flush=True)

import h3_pipeline  # noqa: E402  the one place the pipeline is built (build_check.py runs the same build())

# A ComponentsManager + auto offload, not a plain .to("cuda"): the fl2va transformer partition alone is
# 61.7GB in bf16 and the Qwen3-VL conditioner is another 62.1GB -- together with the two VAEs that does not
# sit resident on one 80GB card at once. The manager keeps the weights in host RAM (220GiB on this replica)
# and moves onto the GPU only what the current step needs, which is diffusers' own documented recipe for a
# single 80GB card (see the module docstring above and README.md's Memory section).
PIPE, MANAGER = h3_pipeline.build(MINIMAX_H3_DIR, reserve=MEMORY_RESERVE_MARGIN)
print(f"[boot] ready in {time.time() - _t0:.1f}s | memory_reserve_margin={MEMORY_RESERVE_MARGIN}", flush=True)


def _load_image(src):
    """data URL / raw base64 / https:// URL -> PIL Image, through safe_fetch's SSRF guard -- job inputs can
    arrive from an LLM agent, so a URL must never be able to reach an internal address. Raises ValueError on
    a refused or unreachable source; the caller turns that into {"error": ...}."""
    if not src:
        return None
    from PIL import Image
    data = safe_fetch.fetch_image_bytes(src)
    return Image.open(io.BytesIO(data)).convert("RGB")


def handler(job):
    job_input = job.get("input") or {}
    prompt = (job_input.get("prompt") or "").strip()
    if not prompt:
        return {"error": "prompt is required"}

    try:
        image = _load_image(job_input.get("image"))
        last_image = _load_image(job_input.get("last_image"))
    except ValueError as e:
        return {"error": f"could not load image: {e}"}

    started = time.time()
    try:
        # Default 5s, clamped to 5..10s (the model's own window is 5-15s; the task's own figure is used
        # here rather than the wider window). 24fps is the model's fixed, non-negotiable rate.
        seconds = min(max(float(job_input.get("seconds", 5)), 5), 10)
        num_frames = round(seconds * FPS)

        kwargs = {
            "prompt": prompt,
            "num_frames": num_frames,
            # Guidance is baked into the checkpoint's weights (distilled): no guidance_scale, no
            # negative_prompt -- passing either would be a silent no-op at best and a TypeError at worst.
            "num_inference_steps": min(max(int(job_input.get("steps", 30)), 1), 50),
            "output": ["videos", "audio", "sampling_rate"],
        }
        if image is not None:
            kwargs["image"] = image
        if last_image is not None:
            kwargs["last_image"] = last_image

        # width/height are optional: left out, the pipeline derives its own canvas from the first
        # keyframe's aspect ratio (or 16:9 with no keyframe). Supplied, both must be multiples of 32.
        if "width" in job_input or "height" in job_input:
            width = min(max(int(job_input.get("width", 1344)), 512), 1344) // 32 * 32
            height = min(max(int(job_input.get("height", 768)), 512), 1344) // 32 * 32
            kwargs["width"] = width
            kwargs["height"] = height
        else:
            width = height = None

        seed = int(job_input.get("seed", -1))
        if seed < 0:
            seed = int.from_bytes(os.urandom(4), "big")
        kwargs["generator"] = torch.Generator().manual_seed(seed)

        results = PIPE(**kwargs)
        videos = results["videos"]
        if not videos:
            return {"error": "pipeline returned no video"}
        frames = videos[0]
        audio = results["audio"][0]
        sampling_rate = results["sampling_rate"]

        from diffusers.utils.export_utils import encode_video
        fd, path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        try:
            encode_video(frames, fps=FPS, output_path=path, audio=audio, audio_sample_rate=sampling_rate)
            data = open(path, "rb").read()
        finally:
            os.unlink(path)

        n = len(frames)
        return {
            "video": "data:video/mp4;base64," + base64.b64encode(data).decode(),
            "seconds": round(n / FPS, 2),
            "frames": n,
            "width": width,
            "height": height,
            "seed": seed,
            "generation_seconds": round(time.time() - started, 2),
            "model": "MiniMaxAI/MiniMax-H3",
            "licence": "MiniMax H3 Community License (Territory excludes EU/UK/South Korea/US; binds "
                       "Outputs; Powered by MiniMax H3)",
        }
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        return {"error": f"CUDA OOM: {e}. Try fewer seconds or a smaller width/height."}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


import runpod  # noqa: E402

runpod.serverless.start({"handler": handler})
