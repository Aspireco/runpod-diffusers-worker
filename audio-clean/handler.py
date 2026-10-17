"""Audio cleanup (ElevenLabs "Voice Isolator" equivalent) for the Azure GPU fleet lane `audio-clean`
(T4, 16 GB). Pulls a voice out of noise/music, removes the voice to keep only the background, or just
denoises. Demucs v4 (MIT, facebookresearch/demucs) does the source separation via its own official
`demucs.api.Separator`; DeepFilterNet 3 (MIT OR Apache-2.0, Rikorose/DeepFilterNet) does the speech
denoising via its own official `df.enhance.init_df`/`enhance`. Both are called directly -- the only code
here is the thin per-job glue (fetch/decode, pick stems, resample back, mux) -- see README.md "Premade
search" and the licence table. Both sets of weights are baked into the image at build time (see
Dockerfile); there is no fetch job for this lane.

mode=isolate_voice (default): Demucs "vocals" stem -> DeepFilterNet denoise.
mode=remove_voice: Demucs "drums"+"bass"+"other" stems summed -- the music/ambience bed.
mode=denoise: DeepFilterNet only, no Demucs.

Input:  audio OR video (data/https URL; exactly one required), mode (default "isolate_voice"),
        max_seconds (default 900).
Output: audio_url (data: wav), mode, duration_seconds, sample_rate, generation_seconds, and -- when the
        input was a video -- video_url (the same video, video stream copied, audio replaced and encoded
        as aac, +faststart).
"""

import base64
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

print("[boot] audio-clean handler starting", flush=True)

import torch  # noqa: E402

print(f"[boot] torch {torch.__version__} cuda={torch.cuda.is_available()}", flush=True)

# torch >= 2.6 flipped torch.load's default weights_only to True. demucs/states.py's load_model() calls
# torch.load(path, 'cpu') with no override (confirmed in source, 2026-10-17), and every .th checkpoint it
# loads pickles an actual class reference (e.g. HDemucs) plus args/kwargs/state -- not weights-only
# tensors -- so it would otherwise raise under torch 2.8's default. Every checkpoint loaded below was
# downloaded and size-checked by this worker's own Dockerfile, not supplied by a caller at runtime, so
# reverting to the pre-2.6 default is the old, safe-for-this-use behaviour, not a new risk -- but the
# patch is only live for these two boot-time loads (both models load once, here, and never again): it is
# restored immediately after, so request-time code -- none of which calls torch.load directly -- still
# sees torch's own secure-by-default weights_only=True.
_torch_load = torch.load


def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)


torch.load = _torch_load_compat

DEMUCS_DIR = Path(os.environ.get("DEMUCS_DIR", "/opt/weights/demucs"))
DEMUCS_MODEL = os.environ.get("DEMUCS_MODEL", "htdemucs")
DFN_DIR = os.environ.get("DFN_DIR", "/opt/weights/DeepFilterNet3")
MODES = {"isolate_voice", "remove_voice", "denoise"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_t0 = time.time()
from demucs.api import Separator  # noqa: E402
from demucs.audio import save_audio as demucs_save_audio  # noqa: E402

SEPARATOR = Separator(model=DEMUCS_MODEL, repo=DEMUCS_DIR, device=DEVICE)
DEMUCS_SR = SEPARATOR.samplerate
print(f"[boot] demucs {DEMUCS_MODEL} ready in {time.time() - _t0:.1f}s on {DEVICE} (sr={DEMUCS_SR})",
      flush=True)

_t0 = time.time()
from df.enhance import enhance, init_df  # noqa: E402
from df.io import load_audio as df_load_audio  # noqa: E402
from df.io import resample as df_resample  # noqa: E402
from df.io import save_audio as df_save_audio  # noqa: E402
from df.model import ModelParams  # noqa: E402

# log_file=None: the official default ("enhance.log") writes to the CWD, not /tmp -- disabled outright
# rather than redirected, since nothing here needs a persistent log file. init_df() returns a 3-tuple
# (model, df_state, suffix) -- proven by a real build_check.py failure (ch2f, 2026-10-17: "expected 4,
# got 3"), not by the docs, which disagreed with the actual installed 0.5.6 source.
DF_MODEL, DF_STATE, _DF_SUFFIX = init_df(model_base_dir=DFN_DIR, config_allow_defaults=True,
                                          log_file=None)
DF_SR = ModelParams().sr
print(f"[boot] DeepFilterNet3 ready in {time.time() - _t0:.1f}s (sr={DF_SR})", flush=True)

torch.load = _torch_load  # both boot-time loads are done; restore torch's own secure default


# ---------------------------------------------------------------------------------------------------------
# Small I/O helpers (same pattern as talk-infinitetalk: download / probe / run a subprocess).
# ---------------------------------------------------------------------------------------------------------
def _load_bytes(src: str, timeout: float = 90) -> bytes:
    if src.startswith("http"):
        with urllib.request.urlopen(src, timeout=timeout) as r:
            return r.read()
    raw = src.split(",", 1)[1] if src.startswith("data:") else src
    return base64.b64decode(raw, validate=True)


def _run(cmd: list) -> None:
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {r.stderr.decode(errors='ignore')[-800:]}")


def _ffprobe_duration(path: str) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        return float(r.stdout.decode().strip())
    except ValueError:
        raise RuntimeError(f"ffprobe could not read a duration: {r.stderr.decode(errors='ignore')[-300:]}")


def _ffprobe_samplerate(path: str) -> int:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate",
         "-of", "default=nw=1:nk=1", path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        return int(r.stdout.decode().strip())
    except ValueError:
        raise RuntimeError(f"ffprobe could not read a sample rate: {r.stderr.decode(errors='ignore')[-300:]}")


# ---------------------------------------------------------------------------------------------------------
# Mode pipelines. Each writes a PCM wav at `orig_sr` to `out_wav`.
# ---------------------------------------------------------------------------------------------------------
def _isolate_voice(src_wav: str, out_wav: str, orig_sr: int, tmp_dir: str) -> None:
    _origin, stems = SEPARATOR.separate_audio_file(Path(src_wav))
    vocals_wav = os.path.join(tmp_dir, "vocals.wav")
    demucs_save_audio(stems["vocals"], vocals_wav, DEMUCS_SR)
    _denoise(vocals_wav, out_wav, orig_sr)


def _remove_voice(src_wav: str, out_wav: str, orig_sr: int, tmp_dir: str) -> None:
    _origin, stems = SEPARATOR.separate_audio_file(Path(src_wav))
    bed = stems["drums"] + stems["bass"] + stems["other"]
    bed_wav = os.path.join(tmp_dir, "bed.wav")
    demucs_save_audio(bed, bed_wav, DEMUCS_SR)
    _run(["ffmpeg", "-y", "-i", bed_wav, "-ar", str(orig_sr), out_wav])


def _denoise(src_wav: str, out_wav: str, orig_sr: int) -> None:
    """Mirrors df.enhance's own official per-file CLI sequence: load at DF_SR -> enhance -> resample
    back to the caller's own rate -> save. Downmixed to mono first: DeepFilterNet3 is a mono speech
    model, and a single cleaned voice track is the expected product here regardless of the source's
    channel count."""
    audio, _meta = df_load_audio(src_wav, sr=DF_SR)
    audio = audio.mean(dim=0, keepdim=True)
    enhanced = enhance(DF_MODEL, DF_STATE, audio, pad=True)
    enhanced = df_resample(enhanced.to("cpu"), DF_SR, orig_sr)
    df_save_audio(out_wav, enhanced, sr=orig_sr, log=False)


def _mux_audio(video_path: str, audio_path: str, duration: float, out_path: str, tmp_dir: str) -> None:
    """Copy video stream, aac audio, +faststart -- the handler contract's exact wording. Explicit -map
    (unlike talk-infinitetalk's own _mux_audio, which muxes onto video it generated with no audio track
    of its own) because video_path here is the caller's ORIGINAL file and usually already has an audio
    stream; without -map, ffmpeg's automatic stream selection is not guaranteed to prefer the new one."""
    cropped = os.path.join(tmp_dir, "audio_cropped.wav")
    _run(["ffmpeg", "-y", "-i", audio_path, "-t", f"{duration:.3f}", cropped])
    _run([
        "ffmpeg", "-y", "-i", video_path, "-i", cropped,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-movflags", "+faststart", "-shortest", out_path,
    ])


def handler(job):
    job_input = job.get("input") or {}
    audio_src = job_input.get("audio")
    video_src = job_input.get("video")
    if bool(audio_src) == bool(video_src):
        return {"error": "exactly one of audio or video is required"}
    mode = str(job_input.get("mode") or "isolate_voice").strip().lower()
    if mode not in MODES:
        return {"error": f"mode must be one of {sorted(MODES)}, got {mode!r}"}
    max_seconds = max(1.0, float(job_input.get("max_seconds", 900)))

    started = time.time()
    tmp_dir = tempfile.mkdtemp(prefix="audioclean_")
    try:
        src = video_src or audio_src
        what = "video" if video_src else "audio"
        try:
            raw_bytes = _load_bytes(src)
        except Exception as e:  # noqa: BLE001
            return {"error": f"could not fetch {what}: {type(e).__name__}: {e}"}
        raw_path = os.path.join(tmp_dir, "input.bin")
        with open(raw_path, "wb") as f:
            f.write(raw_bytes)

        src_wav = os.path.join(tmp_dir, "src.wav")
        try:
            _run(["ffmpeg", "-y", "-i", raw_path, "-vn", src_wav])
        except Exception as e:  # noqa: BLE001
            return {"error": f"could not decode {what}: {type(e).__name__}: {e}"}

        try:
            duration = _ffprobe_duration(src_wav)
            orig_sr = _ffprobe_samplerate(src_wav)
        except Exception as e:  # noqa: BLE001
            return {"error": f"could not read audio: {type(e).__name__}: {e}"}
        if duration > max_seconds + 0.1:
            return {"error": f"audio is {duration:.1f}s, longer than max_seconds={max_seconds:.0f}; "
                              "pass a shorter clip or raise max_seconds"}

        out_wav = os.path.join(tmp_dir, "out.wav")
        extra = {}
        if mode == "remove_voice":
            _remove_voice(src_wav, out_wav, orig_sr, tmp_dir)
            extra["model"] = DEMUCS_MODEL
        elif mode == "isolate_voice":
            _isolate_voice(src_wav, out_wav, orig_sr, tmp_dir)
            extra["model"] = DEMUCS_MODEL
        else:  # denoise
            _denoise(src_wav, out_wav, orig_sr)

        out_duration = _ffprobe_duration(out_wav)
        out_sr = _ffprobe_samplerate(out_wav)
        with open(out_wav, "rb") as f:
            out_bytes = f.read()

        result = {
            "audio_url": "data:audio/wav;base64," + base64.b64encode(out_bytes).decode(),
            "mode": mode,
            "duration_seconds": round(out_duration, 2),
            "sample_rate": out_sr,
            "generation_seconds": round(time.time() - started, 2),
            **extra,
        }

        if video_src:
            out_mp4 = os.path.join(tmp_dir, "out.mp4")
            _mux_audio(raw_path, out_wav, out_duration, out_mp4, tmp_dir)
            with open(out_mp4, "rb") as f:
                result["video_url"] = "data:video/mp4;base64," + base64.b64encode(f.read()).decode()

        return result
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        return {"error": f"CUDA OOM: {e}. Try a shorter clip."}
    except Exception as e:  # noqa: BLE001 - surface the reason rather than dying silently
        return {"error": f"{type(e).__name__}: {e}"}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


import runpod  # noqa: E402

runpod.serverless.start({"handler": handler})
