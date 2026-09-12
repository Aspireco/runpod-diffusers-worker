"""TIER 2 CALIBRATION WORKER -- XTTS v2. Output must NEVER be published.

WHAT THIS IS. Coqui XTTS v2 was the default open voice cloner for roughly two years, and it
is the natural yardstick for "what are we giving up by staying commercially clean?". It is
installed here deliberately, to be measured, and for no other purpose.

WHY IT CAN NEVER BE USED COMMERCIALLY. The weights are under the Coqui Public Model License
(CPML), which is non-commercial. That alone would make this Tier 2. What makes it permanent
is that Coqui the company wound down: there is no longer an entity from whom a commercial
licence could be bought. Most non-commercial models are a purchase order away from being
usable. This one is not, and never will be.

WATCH THE CODE/WEIGHTS SPLIT. The Python package that runs this model (`coqui-tts`, the
maintained idiap fork) is MPL-2.0 -- permissive, commercially fine. The weights it loads are
CPML -- not fine. A permissive library is not evidence of permissive weights, and reading
the pip metadata would tell you the opposite of the truth here. The same split applies to
F5-TTS (MIT code, CC-BY-NC weights).

THE TAGGING CONTRACT. Every byte of audio this worker returns carries its tier inside the
file, written before the WAV is encoded, not bolted on afterwards. If the tagging code
cannot be proven to work, THIS WORKER REFUSES TO BOOT (see the _selfcheck call at import).
That is deliberate: a Tier 2 worker that runs while its tagging is broken would emit audio
indistinguishable from licensed output, which is the one failure this whole tier is built to
prevent. Better a dead endpoint than an untraceable asset.
"""

import base64
import io
import json
import os
import struct
import sys
import time
import urllib.request

# Before any heavy import, so "no logs at all" (died pre-Python) stays distinguishable from
# "printed then stopped" (died during import). Those need opposite fixes.
print("[boot] TIER 2 xtts-v2 handler starting", flush=True)

# ---------------------------------------------------------------------------
# Tier tagging. Kept at the top of the file, dependency-free, and self-checked,
# because everything else here is subordinate to it.
# ---------------------------------------------------------------------------

TAG_KEY = "marketing_os_tier"
TIER = "2"
MODEL_REPO = "coqui/XTTS-v2"
LICENSE_ID = "coqui-public-model-license"
LICENSE_URL = "https://coqui.ai/cpml"
ENDPOINT_NAME = os.environ.get("TIER2_ENDPOINT_NAME", "t2-xtts-v2")


def build_tag(extra=None):
    """The canonical tag payload. Mirrors pipeline/tier_guard.py:build_tag exactly."""
    tag = {
        "ns": "marketing-os",
        "tier": TIER,
        "model": MODEL_REPO,
        "license": LICENSE_ID,
        "license_url": LICENSE_URL,
        "commercial_use": "FORBIDDEN",
        "endpoint": ENDPOINT_NAME,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "warning": "CALIBRATION ONLY -- NOT FOR PUBLICATION OR ANY COMMERCIAL USE",
    }
    if extra:
        tag.update(extra)
    return tag


def _riff_chunk(fourcc: bytes, body: bytes) -> bytes:
    """RIFF chunks are word-aligned; an odd-length body takes a pad byte that is not
    counted in the declared size. Getting this wrong produces a file that some readers
    accept and others reject, which is worse than one that fails everywhere."""
    out = fourcc + struct.pack("<I", len(body)) + body
    return out + (b"\x00" if len(body) % 2 else b"")


def wav_with_tag(pcm_f32, sample_rate: int, tag: dict) -> bytes:
    """Encode float samples to a 16-bit WAV that carries the tier tag in LIST/INFO.

    The tag goes in before the audio data chunk, in the same write. There is no window in
    which untagged audio exists as a file -- which is the difference between tagging at
    generation time and tagging afterwards.
    """
    import numpy as np

    arr = np.asarray(pcm_f32, dtype=np.float32).reshape(-1)
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak > 1.0:
        arr = arr / peak
    pcm16 = (np.clip(arr, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()

    blob = json.dumps(tag, separators=(",", ":")).encode("utf-8")
    # ICMT is the standard comment field; ISFT (software) repeats the marker so a reader
    # that only surfaces one of the two still sees it.
    info = b"INFO"
    info += _riff_chunk(b"ICMT", blob + b"\x00")
    info += _riff_chunk(b"ISFT", f"{TAG_KEY}=tier{TIER}-CALIBRATION-ONLY".encode() + b"\x00")
    info += _riff_chunk(b"ICOP", f"{LICENSE_ID} ({LICENSE_URL}) NON-COMMERCIAL".encode() + b"\x00")

    fmt = struct.pack("<HHIIHH", 1, 1, sample_rate, sample_rate * 2, 2, 16)
    body = b"WAVE" + _riff_chunk(b"LIST", info) + _riff_chunk(b"fmt ", fmt) + _riff_chunk(b"data", pcm16)
    return b"RIFF" + struct.pack("<I", len(body)) + body


def _read_riff_info(data: bytes) -> dict:
    """Independent reader used only to verify what we just wrote. Deliberately NOT shared
    with the writer -- a bug common to both would otherwise verify itself as correct."""
    out = {}
    if not (data.startswith(b"RIFF") and data[8:12] == b"WAVE"):
        return out
    pos = 12
    while pos + 8 <= len(data):
        ctype = data[pos:pos + 4]
        size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if ctype == b"LIST" and body[:4] == b"INFO":
            ip = 4
            while ip + 8 <= len(body):
                k = body[ip:ip + 4].decode("latin-1", "replace")
                isz = struct.unpack("<I", body[ip + 4:ip + 8])[0]
                out[k] = body[ip + 8:ip + 8 + isz].rstrip(b"\x00").decode("utf-8", "replace")
                ip += 8 + isz + (isz & 1)
        pos += 8 + size + (size & 1)
    return out


def _selfcheck():
    """Prove the tag survives a real encode/decode round trip, or refuse to start.

    Runs at import, on CPU, in milliseconds. It is the reason this worker can be trusted to
    never emit an unmarked file: if this raises, the container dies at boot and the endpoint
    never accepts a request.
    """
    import numpy as np
    probe = np.zeros(64, dtype=np.float32)
    tag = build_tag({"selfcheck": True})
    blob = wav_with_tag(probe, 24000, tag)
    back = _read_riff_info(blob)
    if TAG_KEY not in back.get("ISFT", ""):
        raise RuntimeError(f"tier tag did not survive WAV encode; ISFT={back.get('ISFT')!r}")
    parsed = json.loads(back.get("ICMT", "{}"))
    if parsed.get("tier") != "2" or parsed.get("commercial_use") != "FORBIDDEN":
        raise RuntimeError(f"tier tag round-tripped wrong: {parsed}")
    if not blob.startswith(b"RIFF") or b"data" not in blob:
        raise RuntimeError("tagged WAV is not a valid RIFF file")
    print("[boot] tier-2 tagging selfcheck OK", flush=True)


_selfcheck()

# ---------------------------------------------------------------------------
# Model. Everything below this line is ordinary inference.
# ---------------------------------------------------------------------------

import torch  # noqa: E402

print(f"[boot] torch {torch.__version__} cuda={torch.cuda.is_available()}", flush=True)

MODEL_DIR = os.environ.get("XTTS_DIR", "/models/xtts-v2")
DEFAULT_SPEAKER = os.environ.get("DEFAULT_SPEAKER_WAV", "/models/xtts-v2/samples/en_sample.wav")

_t0 = time.time()
from TTS.tts.configs.xtts_config import XttsConfig  # noqa: E402
from TTS.tts.models.xtts import Xtts  # noqa: E402

# Loaded at module import, not per-request: FlashBoot keeps the process warm between jobs,
# so a model loaded here is paid for once per cold start rather than once per generation.
_config = XttsConfig()
_config.load_json(os.path.join(MODEL_DIR, "config.json"))
MODEL = Xtts.init_from_config(_config)
MODEL.load_checkpoint(_config, checkpoint_dir=MODEL_DIR, eval=True, use_deepspeed=False)
if torch.cuda.is_available():
    MODEL.cuda()
SR = int(getattr(_config.audio, "output_sample_rate", 24000))
print(f"[boot] xtts-v2 ready in {time.time() - _t0:.1f}s (sr={SR})", flush=True)


def _fetch_speaker(job_input) -> str:
    """Resolve the reference voice to a local path. URL or base64 both land in /tmp."""
    src = job_input.get("speaker_wav_url") or job_input.get("speaker_wav_base64") or job_input.get("speaker_wav")
    if not src:
        return DEFAULT_SPEAKER
    dst = os.path.join("/tmp", f"spk{int(time.time() * 1000)}.wav")
    if src.startswith("http"):
        with urllib.request.urlopen(src, timeout=90) as r:
            data = r.read()
    else:
        raw = src.split(",", 1)[1] if src.startswith("data:") else src
        data = base64.b64decode(raw, validate=True)
    with open(dst, "wb") as fh:
        fh.write(data)
    return dst


def handler(job):
    job_input = job.get("input") or {}
    text = (job_input.get("text") or job_input.get("prompt") or "").strip()
    if not text:
        return {"error": "text is required"}

    started = time.time()
    try:
        speaker = _fetch_speaker(job_input)
        language = job_input.get("language", "en")

        out = MODEL.synthesize(
            text,
            _config,
            speaker_wav=speaker,
            language=language,
            temperature=float(job_input.get("temperature", 0.65)),
            length_penalty=float(job_input.get("length_penalty", 1.0)),
            repetition_penalty=float(job_input.get("repetition_penalty", 2.0)),
            top_k=int(job_input.get("top_k", 50)),
            top_p=float(job_input.get("top_p", 0.85)),
        )
        wav = out["wav"] if isinstance(out, dict) else out

        tag = build_tag({
            "text_sha1": __import__("hashlib").sha1(text.encode()).hexdigest()[:16],
            "language": language,
        })
        data = wav_with_tag(wav, SR, tag)

        # Verify on the way out as well as at boot. Cheap, and it closes the window where a
        # later code change breaks tagging for real jobs while the boot probe still passes.
        check = _read_riff_info(data)
        if TAG_KEY not in check.get("ISFT", ""):
            return {"error": "REFUSED: tier tag missing from generated audio; nothing returned"}

        import numpy as np
        n = int(np.asarray(wav).reshape(-1).size)
        return {
            "audio_url": "data:audio/wav;base64," + base64.b64encode(data).decode(),
            "tier": TIER,
            "tier_warning": "CALIBRATION ONLY -- NOT FOR PUBLICATION OR ANY COMMERCIAL USE",
            "model": MODEL_REPO,
            "license": LICENSE_ID,
            "license_url": LICENSE_URL,
            "sample_rate": SR,
            "seconds": round(n / SR, 2),
            "generation_seconds": round(time.time() - started, 2),
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


import runpod  # noqa: E402

runpod.serverless.start({"handler": handler})
