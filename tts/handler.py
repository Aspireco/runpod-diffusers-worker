"""Text-to-speech worker: Chatterbox Multilingual (MIT) with zero-shot voice cloning.

Chatterbox clones a voice from a few seconds of reference audio and can then speak it in
23 languages — record the brand voice once in English and generate French from the same
clone, without re-recording. That cross-lingual transfer is the reason it's here rather
than a smaller preset-voice model.

Deliberately NOT run with HF_HUB_OFFLINE: the weights are ~3.5GB, egress works fine on
Runpod (the "outgoing traffic has been disabled" error is that flag, not a network block),
and Chatterbox has a documented bug where it contacts the hub even with a warm cache. So
we let it fetch on cold start and let FlashBoot keep it warm after.

Note every output carries Resemble's inaudible Perth watermark. Harmless for ads, but real.
"""

import base64
import io
import os
import tempfile
import time
import urllib.request

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_t0 = time.time()
from chatterbox.mtl_tts import ChatterboxMultilingualTTS  # noqa: E402

MODEL = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)
print(f"[boot] chatterbox multilingual ready in {time.time() - _t0:.1f}s on {DEVICE}", flush=True)


def _fetch_reference(job_input):
    """Materialise the voice reference to a local file, from a URL or inline base64."""
    ref = job_input.get("voice_url") or job_input.get("voice_base64")
    if not ref:
        return None
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    if ref.startswith("http"):
        urllib.request.urlretrieve(ref, path)
    else:
        raw = ref.split(",", 1)[1] if ref.startswith("data:") else ref
        with open(path, "wb") as f:
            f.write(base64.b64decode(raw))
    return path


def handler(job):
    job_input = job.get("input") or {}
    text = (job_input.get("text") or "").strip()
    if not text:
        return {"error": "text is required"}

    language = job_input.get("language_id") or job_input.get("language") or "en"
    ref_path = None
    started = time.time()

    try:
        ref_path = _fetch_reference(job_input)
        kwargs = {"language_id": language}
        if ref_path:
            kwargs["audio_prompt_path"] = ref_path
        # exaggeration drives emotional intensity; cfg_weight trades fidelity against pace
        if "exaggeration" in job_input:
            kwargs["exaggeration"] = float(job_input["exaggeration"])
        if "cfg_weight" in job_input:
            kwargs["cfg_weight"] = float(job_input["cfg_weight"])

        wav = MODEL.generate(text, **kwargs)

        import torchaudio
        buf = io.BytesIO()
        torchaudio.save(buf, wav.cpu(), MODEL.sr, format="wav")
        audio_b64 = base64.b64encode(buf.getvalue()).decode()

        return {
            "audio_url": "data:audio/wav;base64," + audio_b64,
            "sample_rate": MODEL.sr,
            "language": language,
            "cloned": bool(ref_path),
            "characters": len(text),
            "generation_seconds": round(time.time() - started, 2),
        }
    except Exception as e:  # surface the reason rather than dying silently
        return {"error": f"{type(e).__name__}: {e}"}
    finally:
        if ref_path and os.path.exists(ref_path):
            os.unlink(ref_path)


import runpod  # noqa: E402

runpod.serverless.start({"handler": handler})
