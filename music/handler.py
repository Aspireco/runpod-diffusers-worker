"""Music worker: ACE-Step 1.5 (MIT) for ad beds and background tracks.

ACE-Step is here rather than Stable Audio because Stability's Community Licence terminates
above $1M annual revenue aggregated across affiliates -- with five businesses under one
agency that is a live risk. ACE-Step is MIT with no revenue condition, does vocals in 50+
languages, and renders a full track in seconds on a mid-range card.

Caveat worth knowing: its "licensed / royalty-free" training-data claim appears on the
model card but is not documented in the paper. For paid advertising that is a judgement
call, not a settled fact.
"""

import base64
import io
import os
import tempfile
import time

MODEL_ID = os.environ.get("MODEL_ID", "ACE-Step/ACE-Step-v1-3.5B")
CHECKPOINT_DIR = os.environ.get("ACE_CHECKPOINT_DIR", "/opt/ace-checkpoints")

_t0 = time.time()
from acestep.pipeline_ace_step import ACEStepPipeline  # noqa: E402

PIPE = ACEStepPipeline(
    checkpoint_dir=CHECKPOINT_DIR,
    dtype="bfloat16",
    torch_compile=False,          # compiling costs more on a cold worker than it saves
)
print(f"[boot] ACE-Step ready in {time.time() - _t0:.1f}s", flush=True)


def handler(job):
    job_input = job.get("input") or {}
    tags = (job_input.get("tags") or job_input.get("prompt") or "").strip()
    if not tags:
        return {"error": "tags (or prompt) is required, e.g. 'warm acoustic, upbeat, corporate'"}

    lyrics = job_input.get("lyrics") or "[inst]"      # [inst] = instrumental, the usual case for ads
    duration = float(job_input.get("duration", 30))
    steps = int(job_input.get("infer_step", 27))
    guidance = float(job_input.get("guidance_scale", 15.0))
    seed = int(job_input.get("seed", -1))

    started = time.time()
    out_path = os.path.join(tempfile.gettempdir(), f"ace_{int(time.time()*1000)}.wav")

    try:
        PIPE(
            prompt=tags,
            lyrics=lyrics,
            audio_duration=duration,
            infer_step=steps,
            guidance_scale=guidance,
            manual_seeds=str(seed) if seed >= 0 else None,
            save_path=out_path,
        )
        with open(out_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()
        return {
            "audio_url": "data:audio/wav;base64," + audio_b64,
            "duration": duration,
            "tags": tags,
            "instrumental": lyrics.strip() == "[inst]",
            "generation_seconds": round(time.time() - started, 2),
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    finally:
        if os.path.exists(out_path):
            os.unlink(out_path)


import runpod  # noqa: E402

runpod.serverless.start({"handler": handler})
