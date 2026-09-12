"""Upscale / interpolate worker: Real-ESRGAN (BSD-3-Clause) and RIFE (MIT).

Two operations in one image, selected by the MODE env var, because they are almost always
used together -- interpolate to smooth a clip, upscale to finish it -- and the shared
ffmpeg/tensor plumbing is most of the code either would need on its own. Deploy the same
image twice with MODE=upscale and MODE=interpolate.

Model choices are licence-driven first, quality second:

  upscale      Real-ESRGAN x2plus / x4plus from xinntao's own releases (BSD-3-Clause).
               SeedVR2 is the better restorer and is Apache-2.0, but its only inference
               path wants a hand-built NVIDIA apex wheel, flash-attn and an 80GB card --
               three separate ways to die before the first log line, for a job that is a
               2-4x upscale. 4x-UltraSharp and 4x-Remacri are sharper again and both
               CC-BY-NC-SA, so they are out no matter how good they look.

  interpolate  RIFE v4.26 from hzwer/Practical-RIFE (MIT). GIMM-VFI and the TensorRT RIFE
               ports are non-commercial.

Weights are baked into the image rather than pulled at boot: together they are ~160MB,
which is small enough that baking them beats a cold-start download and removes a runtime
network dependency completely.
"""

import base64
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

MODE = os.environ.get("MODE", "upscale").strip().lower()
if MODE not in ("upscale", "interpolate"):
    # Fail at import so a bad template env var kills the build/boot loudly instead of
    # returning the same confusing error on every job.
    raise SystemExit(f"MODE must be 'upscale' or 'interpolate', got {MODE!r}")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", "/opt/weights")
RIFE_DIR = os.environ.get("RIFE_DIR", "/opt/rife")

# Tiling keeps VRAM flat regardless of input size -- a 4x pass over a full 4K frame in one
# shot needs far more memory than the cards this runs on have.
TILE = int(os.environ.get("TILE", "512"))
TILE_OVERLAP = int(os.environ.get("TILE_OVERLAP", "32"))

MAX_FRAMES = int(os.environ.get("MAX_FRAMES", "1200"))
MAX_PAYLOAD_MB = float(os.environ.get("MAX_PAYLOAD_MB", "18"))
CRF = os.environ.get("X264_CRF", "18")

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True     # frame sizes are constant across a whole clip

_t0 = time.time()

if MODE == "upscale":
    from spandrel import ImageModelDescriptor, ModelLoader  # noqa: E402

    # Both scales load at import. Loading the one the job asks for would be lazier and
    # wrong: FlashBoot snapshots the idle process, so anything loaded inside the handler
    # is paid for again on every cold start. 134MB resident for the pair is cheap.
    # fp32 throughout -- RRDBNet is a known source of fp16 overflow artefacts and black
    # output on some cards, and the tiles are small enough that fp32 is not the bottleneck.
    _loader = ModelLoader(device=torch.device(DEVICE))
    UPSCALERS = {}
    for _factor, _file in ((2, "RealESRGAN_x2plus.pth"), (4, "RealESRGAN_x4plus.pth")):
        _d = _loader.load_from_file(os.path.join(WEIGHTS_DIR, _file))
        if not isinstance(_d, ImageModelDescriptor):
            raise SystemExit(f"{_file} is not an image-to-image model")
        if _d.scale != _factor:
            raise SystemExit(f"{_file} reports scale {_d.scale}, expected {_factor}")
        # Kept as separate statements: spandrel's descriptor is not an nn.Module, so
        # chaining .to().eval() would quietly store None if either stops returning self.
        _d.to(torch.device(DEVICE))
        _d.eval()
        UPSCALERS[_factor] = _d
    print(f"[boot] Real-ESRGAN x2/x4 ready in {time.time() - _t0:.1f}s", flush=True)

else:
    sys.path.insert(0, RIFE_DIR)
    from train_log.IFNet_HDv3 import IFNet  # noqa: E402

    FLOWNET = IFNet()
    _sd = torch.load(os.path.join(RIFE_DIR, "train_log", "flownet.pkl"),
                     map_location="cpu", weights_only=True)
    # The published checkpoint is DDP-saved, so every key carries a "module." prefix, and
    # upstream's loader only strips it when called with rank=-1 (which inference_video.py
    # does, and nothing else does). Combined with strict=False that mistake matches zero
    # keys and silently returns an untrained blur instead of raising -- hence the explicit
    # missing_keys check below. strict=False itself is required: the checkpoint also
    # carries the teacher/caltime tensors, which inference-time IFNet does not define.
    _sd = {k.replace("module.", "", 1): v for k, v in _sd.items() if k.startswith("module.")}
    _res = FLOWNET.load_state_dict(_sd, strict=False)
    if _res.missing_keys:
        raise SystemExit(f"RIFE checkpoint left {len(_res.missing_keys)} tensors "
                         f"uninitialised, e.g. {_res.missing_keys[:3]}")
    FLOWNET.eval().to(DEVICE)
    print(f"[boot] RIFE v4.26 ready in {time.time() - _t0:.1f}s "
          f"({len(_sd)} tensors)", flush=True)


# --------------------------------------------------------------------------- io helpers

def _fetch_bytes(src):
    if src.startswith("http"):
        with urllib.request.urlopen(src, timeout=180) as r:
            return r.read()
    raw = src.split(",", 1)[1] if src.startswith("data:") else src
    return base64.b64decode(raw)


def _source(job_input, *keys):
    for k in keys:
        v = job_input.get(k)
        if v:
            return v
    return None


def _load_image(job_input):
    src = _source(job_input, "image_url", "image_base64", "image")
    if not src:
        return None
    return Image.open(io.BytesIO(_fetch_bytes(src))).convert("RGB")


def _load_video(job_input, work):
    src = _source(job_input, "video_url", "video_base64", "video")
    if not src:
        return None
    path = os.path.join(work, "input.mp4")
    with open(path, "wb") as f:
        f.write(_fetch_bytes(src))
    return path


def _data_uri(blob, mime):
    """Runpod drops oversized job outputs, and base64 inflates a payload by a third.
    Fail with the actual number rather than handing back something the platform bins."""
    mb = len(blob) * 4 / 3 / 1e6
    if mb > MAX_PAYLOAD_MB:
        raise RuntimeError(
            f"output would be {mb:.1f}MB as base64, over MAX_PAYLOAD_MB={MAX_PAYLOAD_MB}. "
            "Shorten the clip, drop the scale, or raise X264_CRF."
        )
    return f"data:{mime};base64," + base64.b64encode(blob).decode()


# ------------------------------------------------------------------------ tensor helpers

def _to_tensor(arr):
    """HxWx3 uint8 -> 1x3xHxW float in [0,1]. astype() also un-shares the ffmpeg pipe
    buffer, which numpy hands back read-only."""
    a = arr.astype(np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def _from_tensor(t):
    """1x3xHxW float -> HxWx3 uint8. contiguous() before the transfer because permute
    only rewrites strides, and both PIL and the raw ffmpeg pipe want packed RGB."""
    t = (t.squeeze(0).clamp(0, 1) * 255).round().byte().permute(1, 2, 0)
    return t.contiguous().cpu().numpy()


def _upscale_tensor(t, factor):
    """Tiled so VRAM tracks tile size, not image size. Each tile is fed to the model with
    an overlap margin and the margin is then discarded, which puts the seams inside the
    model's receptive field instead of on it -- untiled edges show as a visible grid."""
    model = UPSCALERS[factor]
    t = t.to(model.device, model.dtype)
    _, _, h, w = t.shape
    s = model.scale

    if TILE <= 0 or (h <= TILE and w <= TILE):
        with torch.no_grad():
            return model(t)

    out = torch.zeros(1, 3, h * s, w * s, device=model.device, dtype=model.dtype)
    for y in range(0, h, TILE):
        for x in range(0, w, TILE):
            y1, x1 = min(y + TILE, h), min(x + TILE, w)
            py0, px0 = max(y - TILE_OVERLAP, 0), max(x - TILE_OVERLAP, 0)
            py1, px1 = min(y1 + TILE_OVERLAP, h), min(x1 + TILE_OVERLAP, w)
            with torch.no_grad():
                patch = model(t[:, :, py0:py1, px0:px1])
            top, left = (y - py0) * s, (x - px0) * s
            out[:, :, y * s:y1 * s, x * s:x1 * s] = \
                patch[:, :, top:top + (y1 - y) * s, left:left + (x1 - x) * s]
    return out


def _rife(a, b, timesteps, flow_scale):
    """Frames between a and b at the given timesteps (0..1 exclusive).

    The 128px padding is upstream's: IFNet runs a five-level pyramid, so both dimensions
    have to be a multiple of 128/flow_scale or the coarse levels round away real pixels.
    flow_scale below 1.0 computes flow at lower resolution -- 0.5 is upstream's advice for
    4K, where full-resolution flow is both slow and noisier.
    """
    _, _, h, w = a.shape
    step = max(128, int(128 / flow_scale))
    ph = ((h - 1) // step + 1) * step
    pw = ((w - 1) // step + 1) * step
    pad = (0, pw - w, 0, ph - h)
    x = torch.cat((F.pad(a, pad), F.pad(b, pad)), 1)
    scales = [16 / flow_scale, 8 / flow_scale, 4 / flow_scale, 2 / flow_scale, 1 / flow_scale]

    out = []
    with torch.no_grad():
        for ts in timesteps:
            _, _, merged = FLOWNET(x, ts, scales)
            out.append(merged[-1][:, :, :h, :w])
    return out


# ------------------------------------------------------------------------ ffmpeg plumbing

def _run(*args):
    p = subprocess.run(args, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(f"{args[0]} failed: {p.stderr.strip()[-400:]}")
    return p.stdout


def _probe(path):
    j = json.loads(_run("ffprobe", "-v", "error", "-print_format", "json",
                        "-show_streams", "-show_format", path))
    streams = j.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise RuntimeError("input has no video stream")

    fps = 0.0
    for key in ("avg_frame_rate", "r_frame_rate"):
        num, _, den = (video.get(key) or "0/0").partition("/")
        if float(den or 0):
            fps = float(num) / float(den)
        if fps:
            break
    if not fps:
        fps = 30.0

    duration = float(j.get("format", {}).get("duration") or 0) or 0.0
    # nb_frames is absent or wrong on plenty of containers, so the count is estimated for
    # the up-front guard only; the decode loop just reads until the pipe runs dry.
    frames = int(video.get("nb_frames") or 0) or int(round(duration * fps))
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    return int(video["width"]), int(video["height"]), fps, has_audio, frames


def _decode_frames(path, w, h):
    """Yield raw RGB frames off ffmpeg's stdout.

    Streaming rather than dumping a PNG sequence to disk: one 4x-upscaled 1080p frame is
    ~40MB, so a few seconds of footage fills a container disk long before it fills the
    response payload. stderr goes to a real file because a pipe we are not draining can
    deadlock ffmpeg once the kernel buffer fills.
    """
    err = tempfile.TemporaryFile()
    p = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE, stderr=err,
    )
    n = w * h * 3
    try:
        while True:
            buf = p.stdout.read(n)
            if len(buf) < n:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        p.stdout.close()
        p.wait()
        err.seek(0)
        message = err.read().decode("utf-8", "replace").strip()
        err.close()
    if p.returncode:
        raise RuntimeError(f"ffmpeg decode failed: {message[-400:]}")


def _open_encoder(path, w, h, fps, audio_from=None):
    cmd = ["ffmpeg", "-v", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{fps:.6f}",
           "-i", "-"]
    if audio_from:
        cmd += ["-i", audio_from, "-map", "0:v:0", "-map", "1:a:0",
                "-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", CRF,
            # yuv420p cannot represent odd dimensions, and interpolation preserves whatever
            # the source had. Cropping to even is a no-op on everything else.
            "-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
    err = tempfile.TemporaryFile()
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=err), err


def _feed(proc, err, data):
    """An encoder that has already died surfaces as BrokenPipeError on the next write,
    which says nothing about the cause. Swap in ffmpeg's own account of it."""
    try:
        proc.stdin.write(data)
    except BrokenPipeError:
        err.seek(0)
        detail = err.read().decode("utf-8", "replace").strip()[-400:]
        raise RuntimeError(f"ffmpeg encode died: {detail}") from None


def _close_encoder(proc, err):
    proc.stdin.close()
    proc.wait()
    err.seek(0)
    message = err.read().decode("utf-8", "replace").strip()
    err.close()
    if proc.returncode:
        raise RuntimeError(f"ffmpeg encode failed: {message[-400:]}")


# ------------------------------------------------------------------------------- handlers

def _upscale_job(job_input, work):
    scale = int(job_input.get("scale", 2))
    if scale not in UPSCALERS:
        return {"error": f"scale must be 2 or 4, got {scale}"}

    video = _load_video(job_input, work)
    if video:
        w, h, fps, has_audio, frames = _probe(video)
        if frames > MAX_FRAMES:
            return {"error": f"{frames} frames exceeds MAX_FRAMES={MAX_FRAMES}"}

        dst = os.path.join(work, "out.mp4")
        # Duration is untouched by upscaling, so the original audio still lines up.
        enc, err = _open_encoder(dst, w * scale, h * scale, fps, video if has_audio else None)
        written, finished = 0, False
        try:
            for arr in _decode_frames(video, w, h):
                out = _upscale_tensor(_to_tensor(arr).to(DEVICE), scale)
                _feed(enc, err, _from_tensor(out).tobytes())
                written += 1
            _close_encoder(enc, err)
            finished = True
        finally:
            # A half-fed encoder would otherwise outlive the job as a zombie holding a
            # file in the work dir this handler is about to delete.
            if not finished:
                enc.kill()
                enc.wait()
                err.close()

        with open(dst, "rb") as f:
            blob = f.read()
        return {
            "video_url": _data_uri(blob, "video/mp4"),
            "width": w * scale, "height": h * scale, "fps": round(fps, 3),
            "frames": written, "audio": bool(has_audio), "scale": scale,
            "model": f"RealESRGAN_x{scale}plus",
        }

    img = _load_image(job_input)
    if img is None:
        return {"error": "image_url, image_base64, video_url or video_base64 is required"}

    out = _upscale_tensor(_to_tensor(np.asarray(img)).to(DEVICE), scale)
    buf = io.BytesIO()
    Image.fromarray(_from_tensor(out)).save(buf, format="PNG")
    return {
        "image_url": _data_uri(buf.getvalue(), "image/png"),
        "width": img.width * scale, "height": img.height * scale,
        "scale": scale, "model": f"RealESRGAN_x{scale}plus",
    }


def _interpolate_job(job_input, work):
    video = _load_video(job_input, work)
    if not video:
        # Interpolation needs motion between two real frames; a single still has none.
        return {"error": "video_url or video_base64 is required in interpolate mode"}

    multiplier = int(job_input.get("multiplier", 2))
    if multiplier not in (2, 4):
        return {"error": f"multiplier must be 2 or 4, got {multiplier}"}
    # Upstream's documented set. Validated rather than clamped because it is a divisor --
    # flow_scale=0 would be a ZeroDivisionError several frames into the job.
    flow_scale = float(job_input.get("flow_scale", 1.0))
    if flow_scale not in (0.25, 0.5, 1.0, 2.0, 4.0):
        return {"error": f"flow_scale must be one of 0.25/0.5/1.0/2.0/4.0, got {flow_scale}"}

    w, h, in_fps, has_audio, frames = _probe(video)
    if frames > MAX_FRAMES:
        return {"error": f"{frames} frames exceeds MAX_FRAMES={MAX_FRAMES}"}

    out_fps = float(job_input.get("fps") or in_fps * multiplier)
    # Playing n*multiplier frames at anything other than fps*multiplier changes the clip's
    # wall-clock length, at which point the original audio no longer fits it.
    keep_audio = has_audio and abs(out_fps - in_fps * multiplier) < 0.01

    dst = os.path.join(work, "out.mp4")
    enc, err = _open_encoder(dst, w, h, out_fps, video if keep_audio else None)
    steps = [i / multiplier for i in range(1, multiplier)]

    prev, written = None, 0
    for arr in _decode_frames(video, w, h):
        cur = _to_tensor(arr).to(DEVICE)
        if prev is not None:
            for mid in _rife(prev, cur, steps, flow_scale):
                _feed(enc, err, _from_tensor(mid).tobytes())
                written += 1
        # Source frames are passed through as-is rather than round-tripped through the
        # tensor path, so the originals come out bit-identical.
        _feed(enc, err, arr.tobytes())
        written += 1
        prev = cur

    if prev is None:
        enc.kill()
        err.close()
        return {"error": "no frames decoded from the input video"}
    _close_encoder(enc, err)

    with open(dst, "rb") as f:
        blob = f.read()
    return {
        "video_url": _data_uri(blob, "video/mp4"),
        "width": w, "height": h,
        "fps": round(out_fps, 3), "source_fps": round(in_fps, 3),
        "frames": written, "multiplier": multiplier, "audio": bool(keep_audio),
        "model": "RIFE v4.26",
    }


def handler(job):
    job_input = job.get("input") or {}
    started = time.time()
    work = tempfile.mkdtemp(prefix="job_")
    try:
        result = _upscale_job(job_input, work) if MODE == "upscale" \
            else _interpolate_job(job_input, work)
        result["generation_seconds"] = round(time.time() - started, 2)
        return result
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    finally:
        shutil.rmtree(work, ignore_errors=True)


import runpod  # noqa: E402

runpod.serverless.start({"handler": handler})
