"""Serverless image worker that loads its model from Runpod's host-side cache.

No weights are baked into the image and no network volume is attached. Runpod mounts
whatever model the endpoint references at /runpod-volume/huggingface-cache/hub, and this
handler resolves the snapshot directory itself. That keeps the image tiny (so Runpod's
30-minute build cap is never in play), avoids pinning the endpoint to one datacentre,
and gives cold starts measured in seconds rather than minutes.

Set MODEL_ID to the same repo the endpoint's model reference points at.
"""

import base64
import glob
import io
import os
import time

import torch

MODEL_ID = os.environ.get("MODEL_ID", "Tongyi-MAI/Z-Image-Turbo")
CACHE_ROOT = os.environ.get("HF_CACHE_ROOT", "/runpod-volume/huggingface-cache/hub")

# Per-model sampling defaults. Turbo-class models are distilled to a handful of steps and
# expect guidance disabled -- running them at SDXL's 25 steps/CFG 7 produces mush.
# `offload` keeps only the active submodule resident on the GPU: Qwen-Image is 20B, which
# at bf16 needs more than a 48GB card once the text encoder and VAE are alongside it, and
# a straight .to("cuda") OOMs during weight load.
DEFAULTS = {
    "Tongyi-MAI/Z-Image-Turbo": {"steps": 8, "guidance": 0.0, "size": (1024, 1024)},
    "black-forest-labs/FLUX.2-klein-4B": {"steps": 4, "guidance": 0.0, "size": (1024, 1024)},
    "Qwen/Qwen-Image-2512": {"steps": 28, "guidance": 4.0, "size": (1328, 1328), "offload": True},
}
CFG = DEFAULTS.get(MODEL_ID, {"steps": 20, "guidance": 3.5, "size": (1024, 1024)})

# Env var wins, so a lane can be switched to offload without a rebuild.
_env_offload = os.environ.get("CPU_OFFLOAD")
OFFLOAD = (_env_offload == "1") if _env_offload is not None else bool(CFG.get("offload"))


def find_snapshot(model_id: str) -> str:
    """Locate the cached snapshot Runpod mounted for this model.

    The cache mirrors huggingface_hub's layout: models--{org}--{name}/snapshots/{commit}.
    A repo can hold several commits, so take the most recent.
    """
    repo_dir = "models--" + model_id.replace("/", "--")
    pattern = os.path.join(CACHE_ROOT, repo_dir, "snapshots", "*")
    snapshots = [p for p in glob.glob(pattern) if os.path.isdir(p)]
    if not snapshots:
        raise RuntimeError(
            f"No cached snapshot for {model_id} under {CACHE_ROOT}. "
            "Set the endpoint's model reference to this repo and redeploy."
        )
    return max(snapshots, key=os.path.getmtime)


# Load at import, not inside the handler: FlashBoot snapshots the process once it is idle,
# so anything loaded lazily on the first request is re-loaded on every cold start instead
# of being captured in the snapshot.
_t0 = time.time()
_snapshot = find_snapshot(MODEL_ID)
print(f"[boot] {MODEL_ID} -> {_snapshot}", flush=True)

from diffusers import DiffusionPipeline  # noqa: E402  (import after the path check)

PIPE = DiffusionPipeline.from_pretrained(
    _snapshot,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
)
if OFFLOAD:
    # Streams submodules on and off the GPU as the pipeline runs. Slower per image, but it
    # is the difference between running on a 48GB card and OOMing during load.
    PIPE.enable_model_cpu_offload()
    print("[boot] cpu offload enabled", flush=True)
else:
    PIPE.to("cuda")
try:
    PIPE.set_progress_bar_config(disable=True)
except Exception:
    pass
print(f"[boot] ready in {time.time() - _t0:.1f}s", flush=True)


def handler(job):
    job_input = job.get("input") or {}
    prompt = (job_input.get("prompt") or "").strip()
    if not prompt:
        return {"error": "prompt is required"}

    width = int(job_input.get("width") or CFG["size"][0])
    height = int(job_input.get("height") or CFG["size"][1])
    steps = int(job_input.get("num_inference_steps") or CFG["steps"])
    guidance = float(job_input.get("guidance_scale", CFG["guidance"]))
    seed = int(job_input.get("seed", -1))

    # Diffusers needs both dimensions on a multiple of 8 or the VAE decode errors out.
    width -= width % 8
    height -= height % 8

    kwargs = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
    }
    negative = (job_input.get("negative_prompt") or "").strip()
    if negative:
        kwargs["negative_prompt"] = negative
    if seed >= 0:
        kwargs["generator"] = torch.Generator(device="cuda").manual_seed(seed)

    started = time.time()
    try:
        image = PIPE(**kwargs).images[0]
    except TypeError:
        # A few pipelines reject negative_prompt outright; retry without it rather than
        # failing the job over an optional field.
        kwargs.pop("negative_prompt", None)
        image = PIPE(**kwargs).images[0]

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return {
        "image_url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
        "model": MODEL_ID,
        "width": width,
        "height": height,
        "steps": steps,
        "seed": seed,
        "generation_seconds": round(time.time() - started, 2),
    }


import runpod  # noqa: E402

runpod.serverless.start({"handler": handler})
