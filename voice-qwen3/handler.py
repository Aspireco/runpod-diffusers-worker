"""Qwen3-TTS 1.7B CustomVoice (Apache-2.0) WITHOUT ComfyUI, for the Azure GPU fleet lane `voice-qwen3` (T4).

The ComfyUI lane (voice-qwen) never got as far as claiming a job on Azure (2026-10-16 proof). This calls the
qwen_tts package directly, the same call path John's local stack uses (E:/local-ai/loc.py QWEN_CODE):
Qwen3TTSModel.from_pretrained(dir, device_map, dtype) -> generate_custom_voice(text, speaker, instruct).

fp32 by default: a T4 has no fast bf16, and 1.7B parameters at 4 bytes is ~7 GB, well inside its 16 GB.

Input:  text, speaker (ryan, aiden, serena, vivian, uncle_fu, ono_anna, sohee, eric, dylan), instruct (optional
        speaking direction, e.g. "warm and confident").
Output: audio_url (data: wav), sample_rate, speaker, characters, generation_seconds.
"""

import base64
import io
import os
import time

print("[boot] voice-qwen3 handler starting", flush=True)

import torch  # noqa: E402

print(f"[boot] torch {torch.__version__} cuda={torch.cuda.is_available()}", flush=True)

MODEL_DIR = os.environ.get("MODEL_DIR", "/runpod-volume/qwen3-tts/1.7b-customvoice")
DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
DTYPE = DTYPES.get(os.environ.get("DTYPE", "float32"), torch.float32)
SPEAKERS = {"ryan", "aiden", "serena", "vivian", "uncle_fu", "ono_anna", "sohee", "eric", "dylan"}

if not os.path.isfile(os.path.join(MODEL_DIR, "config.json")):
    raise RuntimeError(f"{MODEL_DIR} has no config.json: run the fetch job for lane voice-qwen3 first")

_t0 = time.time()
from qwen_tts import Qwen3TTSModel  # noqa: E402

MODEL = Qwen3TTSModel.from_pretrained(MODEL_DIR, device_map="cuda:0" if torch.cuda.is_available() else "cpu",
                                      dtype=DTYPE)
print(f"[boot] qwen3-tts ready in {time.time() - _t0:.1f}s on {getattr(MODEL, 'device', '?')} ({DTYPE})", flush=True)


def handler(job):
    job_input = job.get("input") or {}
    text = (job_input.get("text") or "").strip()
    if not text:
        return {"error": "text is required"}
    speaker = str(job_input.get("speaker") or "ryan").lower()
    if speaker not in SPEAKERS:
        return {"error": f"unknown speaker {speaker!r}; choose from {', '.join(sorted(SPEAKERS))}"}
    instruct = (job_input.get("instruct") or "").strip() or None
    started = time.time()
    try:
        wavs, sr = MODEL.generate_custom_voice(text=text, speaker=speaker, instruct=instruct)
        audio = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV")
        return {
            "audio_url": "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode(),
            "sample_rate": sr, "speaker": speaker, "characters": len(text),
            "generation_seconds": round(time.time() - started, 2),
        }
    except Exception as e:  # surface the reason rather than dying silently
        return {"error": f"{type(e).__name__}: {e}"}


import runpod  # noqa: E402

runpod.serverless.start({"handler": handler})
