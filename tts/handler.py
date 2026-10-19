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

`mode: "vc"` adds speech-to-speech voice conversion from the SAME chatterbox-tts package
(`chatterbox.vc.ChatterboxVC`, confirmed in the installed 0.1.7 wheel) -- a second model, over
the same ResembleAI/chatterbox repo this worker already pulls from, but a different file
(s3gen.safetensors + conds.pt, fetched via hf_hub_download, vs. the multilingual TTS model's
s3gen.pt + ve.pt + t3_mtl23ls_v2.safetensors via snapshot_download). It loads lazily on the
first VC job, so a worker that only ever gets TTS jobs never pays for it, and then stays warm
like MODEL does. VC output carries the same Perth watermark.
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


def _fetch_to_file(src: str, dest_path: str, timeout: float = 60.0) -> None:
    """A job input is a data: URL or an http(s) URL; write its bytes to dest_path.

    Same helper as workers/lipsync-musetalk/handler.py and workers/voice-qwen3/handler.py, kept
    word-for-word: the `audio` field (mode=vc's speech to convert) isn't voice_url/voice_base64
    shaped, so it needs its own fetch rather than reusing _fetch_reference above.
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


_VC_MODEL = None


def _get_vc_model():
    """Lazy singleton: ChatterboxVC loads only on the first mode=vc job (own s3gen.safetensors +
    conds.pt, ~1.06GB, fetched from the same ResembleAI/chatterbox repo as MODEL), then stays
    warm for the rest of this worker's life, same as MODEL does for TTS jobs."""
    global _VC_MODEL
    if _VC_MODEL is None:
        from chatterbox.vc import ChatterboxVC  # noqa: E402
        _t0 = time.time()
        _VC_MODEL = ChatterboxVC.from_pretrained(device=DEVICE)
        print(f"[boot] chatterbox VC ready in {time.time() - _t0:.1f}s on {DEVICE}", flush=True)
    return _VC_MODEL


def _handle_vc(job_input):
    """mode: "vc" -- convert `audio` into the voice sampled by voice_url/voice_base64."""
    audio_src = (job_input.get("audio") or "").strip()
    if not audio_src:
        return {"error": "audio is required (a data: or https URL of the speech to convert)"}
    if not (job_input.get("voice_url") or job_input.get("voice_base64")):
        return {"error": "voice_url or voice_base64 is required (a sample of the target voice)"}

    started = time.time()
    audio_path = None
    voice_path = None
    try:
        fd, audio_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        _fetch_to_file(audio_src, audio_path)
        voice_path = _fetch_reference(job_input)  # same voice_url/voice_base64 fields as TTS cloning

        vc = _get_vc_model()
        wav = vc.generate(audio_path, target_voice_path=voice_path)

        import torchaudio
        buf = io.BytesIO()
        torchaudio.save(buf, wav.cpu(), vc.sr, format="wav")
        audio_b64 = base64.b64encode(buf.getvalue()).decode()

        return {
            "audio_url": "data:audio/wav;base64," + audio_b64,
            "sample_rate": vc.sr,
            "mode": "vc",
            "generation_seconds": round(time.time() - started, 2),
        }
    except Exception as e:  # surface the reason rather than dying silently
        return {"error": f"{type(e).__name__}: {e}"}
    finally:
        for p in (audio_path, voice_path):
            if p and os.path.exists(p):
                os.unlink(p)


def handler(job):
    job_input = job.get("input") or {}
    if str(job_input.get("mode") or "").strip().lower() == "vc":
        return _handle_vc(job_input)

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
