"""InfiniteTalk talking presenter (Apache-2.0, MeiGen-AI) for the Azure GPU fleet lane `infinitetalk` (A100 80GB).

A photo + an audio track -> a lip-synced talking video. An existing video + a new audio track -> a
re-dubbed talking video (the model walks the source video's own frames forward as it generates, so it
keeps that video's camera movement; a photo has no movement to keep, so it is just reused every chunk).

This calls the official MeiGen-AI/InfiniteTalk code directly (`wan.InfiniteTalkPipeline.generate_infinitetalk`,
vendored at /opt/infinitetalk, see the Dockerfile) rather than reimplementing it. The only code here is the
thin per-job glue generate_infinitetalk.py itself does inline (download/validate inputs, extract one audio
embedding, call the pipeline, mux the result) -- adapted so the ~45GB of models load ONCE at import instead
of once per CLI invocation, matching this file's "load once, serve many jobs" contract.

Weights come from the Azure Files share (/runpod-volume/infinitetalk/...), staged by the fetch jobs for
manifests gpu/manifests/infinitetalk-*.tsv; the image holds none of them. See README.md for the full file
list, the licence table and the "Premade search" section (why this calls the official repo's own pipeline
class instead of hand-rolling the generation loop).

Input:  image (data/https URL, the presenter photo) OR video (data/https URL, re-dub mode) -- exactly one
        of the two is required; audio (data/https URL, required); optional prompt, size ("infinitetalk-480"
        default, or "infinitetalk-720"), steps, seed, max_seconds (default 60), fps (informational only,
        see NATIVE_FPS below).
Output: video_url (data: mp4), duration_seconds, width, height, fps, frames, generation_seconds (+ a few
        informative extras: steps, seed, lora_active, size).
"""

import base64
import os
import shutil
import subprocess
import sys
import tempfile
import time
import types
import urllib.request

print("[boot] talk-infinitetalk handler starting", flush=True)

import torch  # noqa: E402

print(f"[boot] torch {torch.__version__} cuda={torch.cuda.is_available()}", flush=True)

# ---------------------------------------------------------------------------------------------------------
# Paths. One subfolder per fetched HF repo under INFINITETALK_DIR (gpu/manifests/infinitetalk-*.tsv wires
# the fetch jobs at these exact destinations -- see README.md "File list" for the full inventory).
# ---------------------------------------------------------------------------------------------------------
INFINITETALK_DIR = os.environ.get("INFINITETALK_DIR", "/runpod-volume/infinitetalk")
CKPT_DIR = os.environ.get("CKPT_DIR", os.path.join(INFINITETALK_DIR, "wan21-i2v-14b-480p"))
INFINITETALK_CKPT = os.environ.get(
    "INFINITETALK_CKPT", os.path.join(INFINITETALK_DIR, "infinitetalk-ckpt", "single", "infinitetalk.safetensors")
)
WAV2VEC_DIR = os.environ.get("WAV2VEC_DIR", os.path.join(INFINITETALK_DIR, "wav2vec2"))
LORA_PATH = os.environ.get(
    "LORA_PATH",
    os.path.join(INFINITETALK_DIR, "lightx2v-lora", "loras",
                 "Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors"),
)
# LIGHTNING on by default (matches vid-wan-a14b's own convention): the lightx2v step-distill LoRA cuts an
# 81-frame clip from 40 steps x 3 CFG passes to 8 steps x 2 CFG passes -- about 7.5x less DiT compute.
LIGHTNING = os.environ.get("LIGHTNING", "1") == "1"
# An A100 80GB holds the whole pipeline resident (~45GB: 28GB DiT + 11.4GB T5 + 4.8GB CLIP + 0.5GB VAE)
# with plenty of headroom for activations, so offloading between CPU and GPU buys nothing but delay here.
OFFLOAD_MODEL = os.environ.get("OFFLOAD_MODEL", "0") == "1"

# The model's own fixed rate: get_embedding() in the official code aligns the audio embedding to
# `audio_duration * 25` steps, and save_video_ffmpeg() assumes 25fps -- both unconditionally, not a knob.
# A caller-supplied `fps` is accepted (see handler()) but ignored for generation math; using any other
# value would desync lip-sync against the audio. The output always reports the true native rate.
NATIVE_FPS = 25
FRAME_NUM = 81          # the clip bucket size the model was trained at (4n+1); fixed, not a per-job knob.
MOTION_FRAME = 9         # official default/recommended value in every README example.
MIN_SECONDS = (FRAME_NUM + 1) / NATIVE_FPS + 0.05   # strictly more than FRAME_NUM embedding-frames, +margin
DEFAULT_PROMPT = "a person talking to the camera"


def _require(path: str, what: str) -> None:
    if not os.path.exists(path):
        raise RuntimeError(f"{what} missing at {path}: run the fetch jobs for lane infinitetalk")


# Every file listed in gpu/manifests/infinitetalk-wan21-i2v-480p.tsv, infinitetalk-ckpt.tsv and
# infinitetalk-wav2vec2.tsv, checked individually (not just one representative file per directory) --
# the lightx2v LoRA manifest is deliberately NOT checked here; it is optional and degrades gracefully,
# see the LIGHTNING block below.
for _shard in range(1, 8):
    _require(
        os.path.join(CKPT_DIR, f"diffusion_pytorch_model-{_shard:05d}-of-00007.safetensors"),
        f"Wan2.1-I2V-14B-480P DiT shard {_shard}/7",
    )
_require(os.path.join(CKPT_DIR, "config.json"), "Wan2.1-I2V-14B-480P DiT config.json")
_require(os.path.join(CKPT_DIR, "Wan2.1_VAE.pth"), "Wan2.1 VAE")
_require(os.path.join(CKPT_DIR, "models_t5_umt5-xxl-enc-bf16.pth"), "umT5 text encoder")
_require(os.path.join(CKPT_DIR, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"), "CLIP image encoder")
for _tok_file in ("special_tokens_map.json", "spiece.model", "tokenizer.json", "tokenizer_config.json"):
    _require(os.path.join(CKPT_DIR, "google", "umt5-xxl", _tok_file), f"umT5 tokenizer file {_tok_file}")
for _tok_file in ("sentencepiece.bpe.model", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"):
    _require(os.path.join(CKPT_DIR, "xlm-roberta-large", _tok_file), f"xlm-roberta-large tokenizer file {_tok_file}")
_require(INFINITETALK_CKPT, "InfiniteTalk audio-conditioning checkpoint")
_require(os.path.join(WAV2VEC_DIR, "config.json"), "wav2vec2 config")
_require(os.path.join(WAV2VEC_DIR, "preprocessor_config.json"), "wav2vec2 preprocessor config")
_require(os.path.join(WAV2VEC_DIR, "pytorch_model.bin"), "wav2vec2 weights")

sys.path.insert(0, "/opt/infinitetalk")

_t0 = time.time()
import wan  # noqa: E402
from wan.configs import WAN_CONFIGS  # noqa: E402

print(f"[boot] wan package imported in {time.time() - _t0:.1f}s", flush=True)

# The lightx2v LoRA is applied inside InfiniteTalkPipeline's own constructor (it mutates the DiT's weights
# in place -- there is no per-job on/off switch for it once the pipeline is built), so the decision is made
# here, once, before the ~45GB load. A missing or unreadable file degrades to the plain model rather than
# failing boot: checked cheaply (just the safetensors header) before the expensive constructor call.
_lora_dir, _lora_scales = None, None
if LIGHTNING and os.path.isfile(LORA_PATH):
    try:
        from safetensors import safe_open

        with safe_open(LORA_PATH, framework="pt") as _f:
            _n_tensors = len(list(_f.keys()))
        _lora_dir, _lora_scales = [LORA_PATH], [1.0]
        print(f"[boot] lightx2v LoRA at {LORA_PATH} ({_n_tensors} tensors); will apply at scale 1.0", flush=True)
    except Exception as e:  # noqa: BLE001 - a bad LoRA file must not block the plain model from loading
        print(f"[boot] lightx2v LoRA at {LORA_PATH} unreadable ({type(e).__name__}: {e}); "
              "staying on the plain model", flush=True)
elif LIGHTNING:
    print(f"[boot] LIGHTNING=1 but no LoRA at {LORA_PATH}; staying on the plain model", flush=True)

_t0 = time.time()
_cfg = WAN_CONFIGS["infinitetalk-14B"]
PIPE = wan.InfiniteTalkPipeline(
    config=_cfg,
    checkpoint_dir=CKPT_DIR,
    quant_dir=None,
    device_id=0,
    rank=0,
    t5_fsdp=False,
    dit_fsdp=False,
    use_usp=False,
    t5_cpu=False,
    lora_dir=_lora_dir,
    lora_scales=_lora_scales,
    quant=None,
    dit_path=None,
    infinitetalk_dir=INFINITETALK_CKPT,
)
LIGHTNING_ACTIVE = _lora_dir is not None
print(f"[boot] InfiniteTalkPipeline ready in {time.time() - _t0:.1f}s | lightning={LIGHTNING_ACTIVE} "
      f"offload_model={OFFLOAD_MODEL}", flush=True)

# custom_init() in the official generate_infinitetalk.py: the audio encoder runs on CPU (it is small,
# ~380MB, and this keeps every byte of GPU VRAM for the DiT instead of contending with it).
_t0 = time.time()
from transformers import Wav2Vec2FeatureExtractor  # noqa: E402

from src.audio_analysis.wav2vec2 import Wav2Vec2Model  # noqa: E402

WAV2VEC_FE = Wav2Vec2FeatureExtractor.from_pretrained(WAV2VEC_DIR, local_files_only=True)
# eager attention: transformers 4.5x loads wav2vec2 with sdpa, which refuses the output_attentions the official
# forward passes ("not supported when using the attn_implementation set to sdpa", Azure proof 2026-10-18)
WAV2VEC_MODEL = Wav2Vec2Model.from_pretrained(WAV2VEC_DIR, local_files_only=True, attn_implementation="eager").to("cpu")
WAV2VEC_MODEL.feature_extractor._freeze_parameters()
print(f"[boot] wav2vec2 audio encoder ready in {time.time() - _t0:.1f}s (cpu)", flush=True)

# The official README's recipe for "FusioniX or Lightx2v" (4-8 steps): text_guide_scale 1.0, audio_guide_scale
# 2.0, shift 2.0. Plain-model recipe: text_guide_scale 5.0, audio_guide_scale 4.0 (both "optimal" per the
# README), shift 7.0 at 480p / 11.0 at 720p (wan's own _validate_args logic for infinitetalk-14B).
DEFAULT_STEPS = 8 if LIGHTNING_ACTIVE else 40
TEXT_GUIDE_SCALE = 1.0 if LIGHTNING_ACTIVE else 5.0
AUDIO_GUIDE_SCALE = 2.0 if LIGHTNING_ACTIVE else 4.0
SHIFT_LORA = 2.0
COLOR_CORRECTION_STRENGTH = 1.0   # the official CLI's own default (generate_infinitetalk.py argparse)
EXTRA_ARGS = types.SimpleNamespace(use_teacache=False, use_apg=False)

print(f"[boot] ready | default steps={DEFAULT_STEPS} text_guide={TEXT_GUIDE_SCALE} "
      f"audio_guide={AUDIO_GUIDE_SCALE}", flush=True)


# ---------------------------------------------------------------------------------------------------------
# Small helpers -- I/O and the thin audio-prep glue generate_infinitetalk.py does inline (not an exported
# function there, so it is reproduced here; the actual model math is 100% the vendored official code).
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
        raise RuntimeError(f"ffprobe could not read a duration from this audio: {r.stderr.decode(errors='ignore')[-300:]}")


def _loudness_norm(audio_array, sr=16000, lufs=-23):
    """Verbatim from generate_infinitetalk.py: a quiet/silent clip's loudness is undefined, leave it alone."""
    import pyloudnorm as pyln

    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio_array)
    if abs(loudness) > 100:
        return audio_array
    return pyln.normalize.loudness(audio_array, loudness, lufs)


def _get_embedding(speech_array, sr=16000):
    """get_embedding() from generate_infinitetalk.py, against the module-level WAV2VEC_FE/WAV2VEC_MODEL."""
    import numpy as np
    from einops import rearrange

    audio_duration = len(speech_array) / sr
    video_length = audio_duration * NATIVE_FPS
    audio_feature = np.squeeze(WAV2VEC_FE(speech_array, sampling_rate=sr).input_values)
    audio_feature = torch.from_numpy(audio_feature).float().unsqueeze(0)
    with torch.no_grad():
        embeddings = WAV2VEC_MODEL(audio_feature, seq_len=int(video_length), output_hidden_states=True)
    if len(embeddings) == 0:
        return None
    audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
    audio_emb = rearrange(audio_emb, "b s d -> s b d")
    return audio_emb.cpu().detach()


def _tensor_to_u8_frames(video: torch.Tensor):
    """Matches save_video_ffmpeg()'s non-high-quality path exactly: (C,T,H,W) in [-1,1] -> (T,H,W,C) uint8."""
    import numpy as np

    arr = video.permute(1, 2, 3, 0).to(torch.float32).cpu().numpy()
    arr = (arr + 1.0) / 2.0
    arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype("uint8")
    return arr


def _frames_to_mp4(frames, fps: int, out_path: str) -> None:
    """Raw frames piped straight into ffmpeg (same pattern as vid-wan-a14b): no PNG sequence hits disk."""
    h, w = frames.shape[1], frames.shape[2]
    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        out_path,
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    p.stdin.write(frames.tobytes())
    p.stdin.close()
    err = p.stderr.read().decode(errors="ignore")[-800:]
    if p.wait() != 0:
        raise RuntimeError(f"ffmpeg (frames->mp4) failed: {err}")


def _mux_audio(video_path: str, audio_path: str, duration: float, out_path: str, tmp_dir: str) -> None:
    """h264/yuv420p (already true of video_path) + aac, +faststart. Crops audio to the clip's own length
    first (matches save_video_ffmpeg()'s own two-step crop-then-mux), then -shortest as a second safety net."""
    cropped = os.path.join(tmp_dir, "audio_cropped.wav")
    _run(["ffmpeg", "-y", "-i", audio_path, "-t", f"{duration:.3f}", cropped])
    _run([
        "ffmpeg", "-y", "-i", video_path, "-i", cropped,
        "-c:v", "copy", "-c:a", "aac", "-movflags", "+faststart", "-shortest", out_path,
    ])


def handler(job):
    job_input = job.get("input") or {}
    image_src = job_input.get("image")
    video_src = job_input.get("video")
    audio_src = job_input.get("audio")

    if not audio_src:
        return {"error": "audio is required"}
    if bool(image_src) == bool(video_src):
        return {"error": "exactly one of image or video is required"}

    size = job_input.get("size") or "infinitetalk-480"
    if size not in ("infinitetalk-480", "infinitetalk-720"):
        return {"error": f"size must be 'infinitetalk-480' or 'infinitetalk-720', got {size!r}"}
    prompt = (job_input.get("prompt") or "").strip() or DEFAULT_PROMPT
    steps = job_input.get("steps")
    steps = int(steps) if steps is not None else DEFAULT_STEPS
    seed = int(job_input.get("seed", -1))
    max_seconds = max(1.0, float(job_input.get("max_seconds", 60)))
    # `fps` is accepted for forward-compatibility / documentation but never changes the generation math --
    # see the NATIVE_FPS comment above. The output always reports the true rate.
    shift = SHIFT_LORA if LIGHTNING_ACTIVE else (7.0 if size == "infinitetalk-480" else 11.0)

    started = time.time()
    tmp_dir = tempfile.mkdtemp(prefix="infinitetalk_")
    try:
        # ---- audio: download, measure, validate length before spending any GPU time ----
        try:
            audio_bytes = _load_bytes(audio_src)
        except Exception as e:  # noqa: BLE001
            return {"error": f"could not fetch audio: {type(e).__name__}: {e}"}
        audio_raw_path = os.path.join(tmp_dir, "audio_raw.bin")
        with open(audio_raw_path, "wb") as f:
            f.write(audio_bytes)

        try:
            duration = _ffprobe_duration(audio_raw_path)
        except Exception as e:  # noqa: BLE001
            return {"error": f"could not read audio: {type(e).__name__}: {e}"}
        if duration > max_seconds + 0.1:
            return {"error": f"audio is {duration:.1f}s, longer than max_seconds={max_seconds:.0f}; "
                              "pass a shorter clip or raise max_seconds"}
        if duration < MIN_SECONDS:
            return {"error": f"audio is {duration:.2f}s; InfiniteTalk needs at least {MIN_SECONDS:.1f}s "
                              f"(one {FRAME_NUM}-frame clip at {NATIVE_FPS}fps)"}

        # ---- presenter: a static photo, or an existing video to re-dub ----
        if image_src:
            try:
                img_bytes = _load_bytes(image_src)
            except Exception as e:  # noqa: BLE001
                return {"error": f"could not fetch image: {type(e).__name__}: {e}"}
            try:
                from PIL import Image
                import io
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception as e:  # noqa: BLE001
                return {"error": f"could not decode image: {type(e).__name__}: {e}"}
            cond_path = os.path.join(tmp_dir, "cond.png")
            img.save(cond_path)
        else:
            try:
                video_bytes = _load_bytes(video_src)
            except Exception as e:  # noqa: BLE001
                return {"error": f"could not fetch video: {type(e).__name__}: {e}"}
            raw_video_path = os.path.join(tmp_dir, "cond_raw.bin")
            with open(raw_video_path, "wb") as f:
                f.write(video_bytes)
            cond_path = os.path.join(tmp_dir, "cond.mp4")
            try:
                # Re-encode every input to a clean h264/yuv420p mp4 up front, regardless of the source
                # container/codec. The official code only auto-converts av1 -> h264; this is broader and
                # guarantees decord/ffprobe can read whatever the caller actually sent.
                _run(["ffmpeg", "-y", "-i", raw_video_path, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                      "-c:a", "aac", "-movflags", "+faststart", cond_path])
            except Exception as e:  # noqa: BLE001
                return {"error": f"could not decode video: {type(e).__name__}: {e}"}

        # ---- audio prep: normalize to 16kHz mono first (sidesteps librosa's own codec guessing), then
        # the official loudness_norm() + get_embedding() path, saved to a .pt file exactly as the official
        # code does (generate_infinitetalk() reads cond_audio['person1'] as a file path, not a tensor). ----
        norm_wav = os.path.join(tmp_dir, "norm16k.wav")
        try:
            _run(["ffmpeg", "-y", "-i", audio_raw_path, "-ar", "16000", "-ac", "1", norm_wav])
        except Exception as e:  # noqa: BLE001
            return {"error": f"could not decode audio: {type(e).__name__}: {e}"}

        import librosa
        human_speech, _sr = librosa.load(norm_wav, sr=16000)
        human_speech = _loudness_norm(human_speech, sr=16000)
        audio_emb = _get_embedding(human_speech, sr=16000)
        if audio_emb is None:
            return {"error": "failed to extract an audio embedding from this audio"}
        emb_path = os.path.join(tmp_dir, "audio_emb.pt")
        torch.save(audio_emb, emb_path)

        input_clip = {
            "prompt": prompt,
            "cond_video": cond_path,
            "cond_audio": {"person1": emb_path},
        }
        # Derived from the same resampled array fed to _get_embedding (not the separately-ffprobed
        # `duration`), so this matches audio_emb.shape[0] exactly -- generate_infinitetalk() itself takes
        # min(max_frames_num, audio_emb.shape[0]) as its real ceiling either way, so this is a precision
        # improvement, not a correctness one, but it keeps the two numbers from ever disagreeing.
        max_frames_num = max(FRAME_NUM, int((len(human_speech) / 16000) * NATIVE_FPS))

        video = PIPE.generate_infinitetalk(
            input_clip,
            size_buckget=size,
            motion_frame=MOTION_FRAME,
            frame_num=FRAME_NUM,
            shift=shift,
            sampling_steps=steps,
            text_guide_scale=TEXT_GUIDE_SCALE,
            audio_guide_scale=AUDIO_GUIDE_SCALE,
            seed=seed,
            offload_model=OFFLOAD_MODEL,
            max_frames_num=max_frames_num,
            color_correction_strength=COLOR_CORRECTION_STRENGTH,
            extra_args=EXTRA_ARGS,
        )
        if video is None:
            return {"error": "generate_infinitetalk returned no video"}

        frames_u8 = _tensor_to_u8_frames(video)
        n_frames, h, w = frames_u8.shape[0], frames_u8.shape[1], frames_u8.shape[2]

        raw_mp4 = os.path.join(tmp_dir, "raw.mp4")
        _frames_to_mp4(frames_u8, NATIVE_FPS, raw_mp4)

        out_mp4 = os.path.join(tmp_dir, "out.mp4")
        clip_seconds = n_frames / NATIVE_FPS
        _mux_audio(raw_mp4, audio_raw_path, clip_seconds, out_mp4, tmp_dir)

        with open(out_mp4, "rb") as f:
            data = f.read()

        return {
            "video_url": "data:video/mp4;base64," + base64.b64encode(data).decode(),
            "duration_seconds": round(clip_seconds, 2),
            "width": w,
            "height": h,
            "fps": float(NATIVE_FPS),
            "frames": n_frames,
            "generation_seconds": round(time.time() - started, 2),
            "steps": steps,
            "seed": seed,
            "lora_active": LIGHTNING_ACTIVE,
            "size": size,
        }
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        return {"error": f"CUDA OOM: {e}. Try size=infinitetalk-480, fewer steps, or OFFLOAD_MODEL=1."}
    except Exception as e:  # noqa: BLE001 - surface the reason rather than dying silently
        return {"error": f"{type(e).__name__}: {e}"}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


import runpod  # noqa: E402

runpod.serverless.start({"handler": handler})
