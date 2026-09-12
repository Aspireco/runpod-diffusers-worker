"""Transcription worker: faster-whisper (MIT) on CTranslate2.

Chosen over WhisperX deliberately. WhisperX fetches four separate things at inference —
the checkpoint, a *per-language* wav2vec2 alignment model, Silero VAD, and gated pyannote
diarization — which is why it crash-looped twice here. faster-whisper loads one local
directory and nothing else, and int8 puts large-v3 in about 1.5GB.

Word-level timestamps come from Whisper itself, so there's no second model to stage.
"""

import base64
import os
import tempfile
import time
import urllib.request

MODEL_SIZE = os.environ.get("WHISPER_MODEL", "large-v3")
COMPUTE = os.environ.get("WHISPER_COMPUTE", "int8_float16")

_t0 = time.time()
from faster_whisper import WhisperModel  # noqa: E402

MODEL = WhisperModel(MODEL_SIZE, device="cuda", compute_type=COMPUTE)
print(f"[boot] faster-whisper {MODEL_SIZE} ({COMPUTE}) ready in {time.time() - _t0:.1f}s", flush=True)


def _fetch_audio(job_input):
    src = job_input.get("audio_url") or job_input.get("audio_base64") or job_input.get("audio_file")
    if not src:
        return None
    fd, path = tempfile.mkstemp(suffix=".audio")
    os.close(fd)
    if src.startswith("http"):
        urllib.request.urlretrieve(src, path)
    else:
        raw = src.split(",", 1)[1] if src.startswith("data:") else src
        with open(path, "wb") as f:
            f.write(base64.b64decode(raw))
    return path


def handler(job):
    job_input = job.get("input") or {}
    path = None
    started = time.time()

    try:
        path = _fetch_audio(job_input)
        if not path:
            return {"error": "audio_url or audio_base64 is required"}

        segments, info = MODEL.transcribe(
            path,
            language=job_input.get("language"),          # None = auto-detect
            beam_size=int(job_input.get("beam_size", 5)),
            vad_filter=bool(job_input.get("vad_filter", True)),
            word_timestamps=bool(job_input.get("word_timestamps", True)),
        )

        out, text_parts = [], []
        for s in segments:                                # generator — consuming it runs the work
            text_parts.append(s.text)
            seg = {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
            if s.words:
                seg["words"] = [
                    {"word": w.word, "start": round(w.start, 2), "end": round(w.end, 2)}
                    for w in s.words
                ]
            out.append(seg)

        return {
            "text": "".join(text_parts).strip(),
            "segments": out,
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
            "duration": round(info.duration, 2),
            "generation_seconds": round(time.time() - started, 2),
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    finally:
        if path and os.path.exists(path):
            os.unlink(path)


import runpod  # noqa: E402

runpod.serverless.start({"handler": handler})
