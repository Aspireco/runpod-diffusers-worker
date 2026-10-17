"""Qwen3-TTS 1.7B -- CustomVoice, VoiceDesign and Base (cloning) -- WITHOUT ComfyUI, for the Azure
GPU fleet (T4). One image, three lanes: which weights and call this handler makes is chosen by env
QWEN_MODE (custom | design | clone; default custom, so the existing lane `voice-qwen3` is unchanged).

The ComfyUI lane (voice-qwen) never got as far as claiming a job on Azure (2026-10-16 proof). This
calls the qwen_tts package directly, the same call path John's local stack uses
(E:/local-ai/loc.py QWEN_CODE, read 2026-10-17):
  custom: Qwen3TTSModel.from_pretrained(dir, device_map, dtype).generate_custom_voice(text, speaker, instruct)
  design: ...generate_voice_design(text=text, instruct=instruct)
  clone:  ...generate_voice_clone(text=text, ref_audio=<path>, ref_text=ref_text) or, with no
          transcript, generate_voice_clone(text=text, ref_audio=<path>, x_vector_only_mode=True) --
          the installed qwen_tts 0.1.1 (inference/qwen3_tts_model.py) raises "ref_text is required
          when x_vector_only_mode=False" otherwise. generate_voice_clone also requires a Base
          checkpoint (tts_model_type == "base"); CustomVoice/VoiceDesign checkpoints reject it.

fp32 by default: a T4 has no fast bf16, and 1.7B parameters at 4 bytes is ~7 GB, well inside its 16 GB.
Each mode's weights live in their own MODEL_DIR on the Azure Files share, staged by that lane's own
fetch job (gpu/manifests/voice-qwen3.tsv, voice-design.tsv, voice-clone-qwen.tsv); the image holds none.

Input (custom): text, speaker (ryan, aiden, serena, vivian, uncle_fu, ono_anna, sohee, eric, dylan),
                instruct (optional speaking direction, e.g. "warm and confident").
Input (design): text, instruct (required -- a description of the voice to invent, e.g. "a warm,
                confident woman in her thirties, slight Canadian accent, speaking clearly").
Input (clone):  text, ref_audio (required -- data: or https URL of a 5-30s sample), ref_text
                (optional transcript of ref_audio; without it, cloning falls back to
                speaker-vector-only mode, which is weaker but needs no transcript).
Output: audio_url (data: wav), sample_rate, mode, characters, generation_seconds (+ speaker, only
        for custom -- unchanged from before modes existed).
"""

import base64
import io
import os
import tempfile
import time
import urllib.request

print("[boot] voice-qwen3 handler starting", flush=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402

print(f"[boot] torch {torch.__version__} cuda={torch.cuda.is_available()}", flush=True)

QWEN_MODE = (os.environ.get("QWEN_MODE") or "custom").strip().lower()
if QWEN_MODE not in {"custom", "design", "clone"}:
    raise RuntimeError(f"QWEN_MODE must be custom, design or clone, got {QWEN_MODE!r}")

# Per mode: the MODEL_DIR this lane defaults to when the lane's own env doesn't set one, and the
# lane name to name in the cold-cache error below (matches the fetch job / manifest for that mode).
_MODE_DEFAULTS = {
    "custom": ("/runpod-volume/qwen3-tts/1.7b-customvoice", "voice-qwen3"),
    "design": ("/runpod-volume/qwen3-tts/1.7b-voicedesign", "voice-design"),
    "clone": ("/runpod-volume/qwen3-tts/1.7b-base", "voice-clone-qwen"),
}
_DEFAULT_MODEL_DIR, _LANE = _MODE_DEFAULTS[QWEN_MODE]
MODEL_DIR = os.environ.get("MODEL_DIR") or _DEFAULT_MODEL_DIR
DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
DTYPE = DTYPES.get(os.environ.get("DTYPE", "float32"), torch.float32)
SPEAKERS = {"ryan", "aiden", "serena", "vivian", "uncle_fu", "ono_anna", "sohee", "eric", "dylan"}

if not os.path.isfile(os.path.join(MODEL_DIR, "config.json")):
    raise RuntimeError(f"{MODEL_DIR} has no config.json: run the fetch job for lane {_LANE} first")

_t0 = time.time()
from qwen_tts import Qwen3TTSModel  # noqa: E402

MODEL = Qwen3TTSModel.from_pretrained(MODEL_DIR, device_map="cuda:0" if torch.cuda.is_available() else "cpu",
                                      dtype=DTYPE)
print(f"[boot] qwen3-tts {QWEN_MODE} ready in {time.time() - _t0:.1f}s on {getattr(MODEL, 'device', '?')} ({DTYPE})",
      flush=True)


def _fetch_to_file(src: str, dest_path: str, timeout: float = 60.0) -> None:
    """A job input is a data: URL or an http(s) URL; write its bytes to dest_path.

    Same helper as workers/lipsync-musetalk/handler.py, kept word-for-word for the fleet's
    convention: job inputs are never written under /runpod-volume (the Azure Files share is
    read-only in practice), always under /tmp.
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


def _write_wav(wavs, sr) -> str:
    audio = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
    audio = np.asarray(audio).squeeze()  # generate_voice_design/_clone can add a leading dim
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()


def handler(job):
    job_input = job.get("input") or {}
    text = (job_input.get("text") or "").strip()
    if not text:
        return {"error": "text is required"}
    started = time.time()
    ref_path = None
    try:
        extra = {}
        if QWEN_MODE == "custom":
            speaker = str(job_input.get("speaker") or "ryan").lower()
            if speaker not in SPEAKERS:
                return {"error": f"unknown speaker {speaker!r}; choose from {', '.join(sorted(SPEAKERS))}"}
            instruct = (job_input.get("instruct") or "").strip() or None
            wavs, sr = MODEL.generate_custom_voice(text=text, speaker=speaker, instruct=instruct)
            extra = {"speaker": speaker}
        elif QWEN_MODE == "design":
            instruct = (job_input.get("instruct") or "").strip()
            if not instruct:
                return {"error": "instruct is required (a description of the voice to design)"}
            wavs, sr = MODEL.generate_voice_design(text=text, instruct=instruct)
        else:  # clone
            ref_audio = (job_input.get("ref_audio") or "").strip()
            if not ref_audio:
                return {"error": "ref_audio is required (a data: or https URL of a 5-30s sample)"}
            ref_text = (job_input.get("ref_text") or "").strip() or None
            fd, ref_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            _fetch_to_file(ref_audio, ref_path)
            clone_kwargs = {"text": text, "ref_audio": ref_path}
            if ref_text:
                clone_kwargs["ref_text"] = ref_text
            else:
                clone_kwargs["x_vector_only_mode"] = True
            wavs, sr = MODEL.generate_voice_clone(**clone_kwargs)
        return {
            "audio_url": _write_wav(wavs, sr),
            "sample_rate": sr, "mode": QWEN_MODE, "characters": len(text),
            "generation_seconds": round(time.time() - started, 2),
            **extra,
        }
    except Exception as e:  # surface the reason rather than dying silently
        return {"error": f"{type(e).__name__}: {e}"}
    finally:
        if ref_path and os.path.exists(ref_path):
            os.unlink(ref_path)


import runpod  # noqa: E402

runpod.serverless.start({"handler": handler})
