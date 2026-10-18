"""Music worker: ACE-Step 1.5 (MIT, github.com/ace-step/ACE-Step-1.5), DiT-only turbo config, for ad
beds and background tracks.

ACE-Step is here rather than Stable Audio because Stability's Community Licence terminates above
$1M annual revenue aggregated across affiliates -- with five businesses under one agency that is a
live risk. ACE-Step 1.5 is MIT with no revenue condition, does vocals in 50+ languages, and renders
a full track in seconds on a mid-range card. This replaces the earlier v1 (ACE-Step/ACE-Step-v1-3.5B)
integration on this same lane: v1.5 is a different, much larger project (its own FastAPI server,
Gradio UI, LoRA/full training, an optional 5Hz language-model "thinking" stage) built around two
entry points this handler uses directly -- `AceStepHandler` (DiT + VAE + text-encoder) and
`generate_music` (`acestep.inference`).

DiT-only, no LM, on purpose. ACE-Step 1.5's own GPU_COMPATIBILITY.md lists "DiT-only mode without
LM initialization" as the documented low-VRAM path, and reading `generate_music.py` directly shows
`use_lm = (params.thinking or need_lm_for_cot) and llm_handler is not None and llm_handler.llm_initialized
and not skip_lm` -- passing `llm_handler=None` with `thinking=False` and the three `use_cot_*` flags
off guarantees the LM branch never runs (the only other `llm_handler.` access inside `generate_music`
sits behind that same `if use_lm:` guard -- checked by reading the function directly, not assumed).
The 5Hz LM's job is "thinking": turning a short query into a full caption/lyrics/metadata set via
chain-of-thought. A caller here already supplies tags and lyrics directly (this lane's long-standing
contract), so there is nothing for it to do. Skipping it also means never installing nano-vllm (not
on PyPI at all -- see Dockerfile) or a vLLM/pt LM backend, and keeps resident weights to DiT (~4.8GB)
+ VAE (~0.3GB) + the Qwen3-Embedding-0.6B text encoder (~1.2GB) -- comfortably inside a T4's 16GB
with no CPU offload needed.

Qwen3-Embedding-0.6B is NOT part of the optional LM -- reading init_service_loader_components.py
directly shows `_load_text_encoder_and_tokenizer` always loads it from `<checkpoint_dir>/
Qwen3-Embedding-0.6B`, and `conditioning_text.py`'s `_prepare_text_conditioning_inputs` is what
actually tokenizes `caption`/`lyrics` with it -- it is the DiT's own text encoder, required for
every task type, LM or no LM.

The `acestep-5Hz-lm-1.7B` component is staged on the share (see README.md "Weights") purely because
`check_main_model_exists()` (`acestep/model_downloader.py`) requires every file in
`MAIN_MODEL_COMPONENTS` to be present before `initialize_service()` will skip a live HuggingFace
download attempt -- the 1.7B LM is never imported, loaded onto the GPU, or used by this handler; its
~3.7GB on the share is the price of a reliable no-network cold start, not a VRAM cost.

Caveat worth knowing (carried over from the old v1 lane): the "licensed / royalty-free" training-data
claim appears on the model card but is not documented in the paper. For paid advertising that is a
judgement call, not a settled fact.
"""

import base64
import os
import shutil
import tempfile
import time
import urllib.request

CHECKPOINT_DIR = os.environ.get("ACESTEP_CHECKPOINTS_DIR", "/runpod-volume/acestep-v15/checkpoints")
CONFIG_PATH = os.environ.get("ACESTEP_CONFIG_PATH", "acestep-v15-turbo")

# Per-config defaults, same shape as workers/handler.py's own DEFAULTS/CFG for the diffusers lanes.
# `shift=3.0` is turbo's own documented recommendation (INFERENCE.md: "Not auto-corrected ... set
# shift=3.0 for turbo"); a non-turbo config wants the dataclass default of 1.0 instead.
_DEFAULTS = {
    "acestep-v15-turbo": {"steps": 8, "shift": 3.0},
    "acestep-v15-xl-turbo": {"steps": 8, "shift": 3.0},
}
CFG = _DEFAULTS.get(CONFIG_PATH, {"steps": 32, "shift": 1.0})

_t0 = time.time()
import torch  # noqa: E402

from acestep.handler import AceStepHandler  # noqa: E402
from acestep.inference import GenerationConfig, GenerationParams, generate_music  # noqa: E402

# Load at import, not inside the handler, so this runs once per replica rather than once per job
# (same convention as every other worker in this repo -- azure_runner.py's load() runs this whole
# file exactly once before taking jobs off the queue).
DIT = AceStepHandler()
_status, _ok = DIT.initialize_service(
    project_root=None,            # ACESTEP_CHECKPOINTS_DIR above wins over project_root/checkpoints
    config_path=CONFIG_PATH,
    device="cuda" if torch.cuda.is_available() else "cpu",
    use_flash_attention=False,    # let it pick sdpa/eager itself; avoid a flash-attn build entirely
    compile_model=False,          # compiling costs more on a cold worker than it saves
    offload_to_cpu=False,         # no LM competing for VRAM -- DiT+VAE+text-encoder fit a T4 whole
    offload_dit_to_cpu=False,
    quantization=None,            # skip torchao entirely; not needed at this model size
    use_mlx_dit=False,            # this lane is Linux/CUDA only; MLX is Apple Silicon
)
if not _ok:
    raise RuntimeError(f"ACE-Step 1.5 failed to initialize against {CHECKPOINT_DIR}: {_status}")
print(f"[boot] {_status} in {time.time() - _t0:.1f}s", flush=True)


def _fetch_to_file(src: str, dest_path: str, timeout: float = 90.0) -> None:
    """A job input is a data: URL or an http(s) URL; write its bytes to dest_path.

    Same helper as workers/tts/handler.py, workers/lipsync-musetalk/handler.py and
    workers/voice-qwen3/handler.py, kept word-for-word.
    """
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


def handler(job):
    job_input = job.get("input") or {}
    tags = (job_input.get("tags") or job_input.get("prompt") or job_input.get("caption") or "").strip()
    if not tags:
        return {"error": "tags (or prompt/caption) is required, e.g. 'warm acoustic, upbeat, corporate'"}

    lyrics = (job_input.get("lyrics") or "").strip()
    # "[inst]" is the old v1 lane's instrumental sentinel; "[Instrumental]" is v1.5's documented one.
    # Treat both (and blank) as instrumental so existing callers keep working unchanged.
    instrumental = bool(job_input.get("instrumental", lyrics.lower() in ("", "[inst]", "[instrumental]")))
    if instrumental:
        lyrics = "[Instrumental]"

    duration = float(job_input.get("duration", 30))
    steps = int(job_input.get("infer_step", job_input.get("num_inference_steps", CFG["steps"])))
    guidance = float(job_input.get("guidance_scale", 7.0))  # turbo ignores this and forces 1.0 itself
    seed = int(job_input.get("seed", -1))
    language = str(job_input.get("language") or job_input.get("vocal_language") or "unknown")

    # Optional style reference -- ACE-Step 1.5's `reference_audio` is a general GenerationParams
    # field (not gated to a particular task_type), so a plain text2music job can use one too.
    reference_src = job_input.get("reference_audio") or job_input.get("reference_audio_url")
    reference_path = None
    tmp_dir = tempfile.mkdtemp(prefix="acestep_")
    try:
        if reference_src:
            reference_path = os.path.join(tmp_dir, "reference.wav")
            try:
                _fetch_to_file(reference_src, reference_path)
            except Exception as e:  # noqa: BLE001
                return {"error": f"could not fetch reference_audio: {type(e).__name__}: {e}"}

        params = GenerationParams(
            task_type="text2music",
            caption=tags,
            lyrics=lyrics,
            instrumental=instrumental,
            vocal_language=language,
            duration=duration,
            inference_steps=steps,
            guidance_scale=guidance,
            shift=CFG["shift"],
            seed=seed,
            thinking=False,          # no 5Hz LM on this lane -- see module docstring
            use_cot_metas=False,
            use_cot_caption=False,
            use_cot_language=False,
            reference_audio=reference_path,
        )
        config = GenerationConfig(batch_size=1, use_random_seed=(seed < 0), audio_format="wav")

        started = time.time()
        result = generate_music(DIT, None, params, config, save_dir=tmp_dir)
        if not result.success or not result.audios:
            return {"error": result.error or "generation produced no audio"}

        audio = result.audios[0]
        with open(audio["path"], "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()
        actual_duration = audio["tensor"].shape[-1] / audio["sample_rate"]

        return {
            "audio_url": "data:audio/wav;base64," + audio_b64,
            "duration": round(actual_duration, 2),
            "sample_rate": audio["sample_rate"],
            "tags": tags,
            "instrumental": instrumental,
            "model": CONFIG_PATH,
            "seed": audio.get("params", {}).get("seed", seed),
            "generation_seconds": round(time.time() - started, 2),
        }
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        return {"error": f"CUDA OOM: {e}. Try a shorter duration."}
    except Exception as e:  # noqa: BLE001 - surface the reason rather than dying silently
        return {"error": f"{type(e).__name__}: {e}"}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


import runpod  # noqa: E402

runpod.serverless.start({"handler": handler})
