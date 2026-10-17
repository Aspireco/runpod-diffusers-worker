"""Wan2.2-Animate-14B (Apache-2.0): a character image + a driving video -> the character performs
the video's motion ("animate", on the character's own background) or replaces the person inside the
video ("replace"). Azure GPU fleet lane `wan-animate` (A100 80GB PCIe).

Uses the official Wan-Video/Wan2.2 code directly, pinned at commit 1ea34ff48f87168174e12956e200b1d908b1c5ff
(see Dockerfile / README.md): the same two-stage pipeline `generate.py --task animate-14B` runs --
`wan.modules.animate.preprocess.ProcessPipeline` (person detection + ViTPose + optional SAM2 mask
propagation) followed by `wan.WanAnimate.generate()` -- called through their Python API exactly as
vid-wan-a14b/handler.py calls diffusers' pipeline classes, rather than shelling out to generate.py.

ONE substitution from the official code: the bundled person detector (process_checkpoint/det/
yolov10m.onnx, Ultralytics YOLOv10, AGPL-3.0) is replaced by detector_adapter.PersonDetector
(torchvision Faster R-CNN v2, BSD-3-Clause) via a monkey-patch of `pose2d.Yolo` -- see
detector_adapter.py's module docstring and README.md's licence table. Nothing else about the model
or its preprocessing is reimplemented.

This process serves a QUEUE of jobs (the Azure runner shim calls handler() once per job, see
gpu/azure_runner.py), not the one-shot CLI generate.py was written for. That matters for exactly one
thing: generate.py loads a FRESH process per task (animate OR replace, never both), so
`use_relighting_lora` is a constructor-time choice there. Here, loading the model twice to keep both
modes resident would roughly double the ~28GB noise-model footprint for no benefit, so the LoRA is
added once at boot and toggled per job with diffusers' own PeftAdapterMixin.enable_adapters() /
disable_adapters() on the noise model -- the official code never has to do this because it never
serves both modes from one process. See the README's "Serving both modes from one process" note.
"""

import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

print("[boot] wan-animate handler starting", flush=True)

import torch  # noqa: E402

print(f"[boot] torch {torch.__version__} cuda={torch.cuda.is_available()}", flush=True)

WAN_ANIMATE_DIR = os.environ.get("WAN_ANIMATE_DIR", "/runpod-volume/wan-animate")
T5_CPU = os.environ.get("T5_CPU", "1") == "1"  # umt5-xxl encoder on CPU: saves ~11GB VRAM; the
                                                 # prompt is tiny and encoded once per job either way.
DEFAULT_RESOLUTION_AREA = [1280, 720]           # see _parse_resolution: only the PRODUCT matters to
                                                 # the official preprocessing, not this ordering.

# One representative file per component the handler actually touches, not all 411 manifest entries
# (gpu/manifests/wan-animate.tsv + wan-animate-replace.tsv list the exact set; this is a fast boot
# check, not a full integrity scan).
_REQUIRED = [
    "config.json",
    "diffusion_pytorch_model.safetensors.index.json",
    "Wan2.1_VAE.pth",
    "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "relighting_lora.ckpt",
    os.path.join("google", "umt5-xxl", "tokenizer_config.json"),
    os.path.join("xlm-roberta-large", "sentencepiece.bpe.model"),
    os.path.join("process_checkpoint", "pose2d", "vitpose_h_wholebody.onnx", "end2end.onnx"),
    os.path.join("process_checkpoint", "sam2", "sam2_hiera_large.pt"),
]

_t0 = time.time()
for _rel in _REQUIRED:
    _full = os.path.join(WAN_ANIMATE_DIR, _rel)
    if not os.path.isfile(_full):
        raise RuntimeError(
            f"{_full} is missing: run the fetch job for lane wan-animate "
            f"(gpu/manifests/wan-animate.tsv and wan-animate-replace.tsv) before starting this worker."
        )
print(f"[boot] weights present at {WAN_ANIMATE_DIR}", flush=True)

# ---- the one licence fix: patch the AGPL detector out before the preprocessing pipeline is built.
import wan  # noqa: E402  (pip-installed from the pinned commit; see Dockerfile)

PREPROCESS_DIR = os.path.join(os.path.dirname(wan.__file__), "modules", "animate", "preprocess")
sys.path.insert(0, PREPROCESS_DIR)                              # for pose2d.py's own flat imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # our detector_adapter.py

import pose2d  # noqa: E402  the vendored module we monkey-patch, not anything of ours

import detector_adapter  # noqa: E402

pose2d.Yolo = detector_adapter.PersonDetector
print("[boot] AGPL yolov10m.onnx detector replaced with torchvision Faster R-CNN v2 (BSD-3-Clause)",
      flush=True)

from process_pipepline import ProcessPipeline  # noqa: E402
from wan.configs import WAN_CONFIGS  # noqa: E402

CFG = WAN_CONFIGS["animate-14B"]

print("[boot] loading WanAnimate (noise model + T5 + CLIP + VAE + relighting LoRA)...", flush=True)
ANIMATE = wan.WanAnimate(
    config=CFG,
    checkpoint_dir=WAN_ANIMATE_DIR,
    device_id=0,
    rank=0,
    t5_fsdp=False,
    dit_fsdp=False,
    use_sp=False,
    t5_cpu=T5_CPU,
    convert_model_dtype=False,
    use_relighting_lora=True,  # added once; see module docstring for why this is toggled per job
                               # with enable_adapters()/disable_adapters() instead of a second model.
)
_HAS_ADAPTER_TOGGLE = hasattr(ANIMATE.noise_model, "disable_adapters") and hasattr(
    ANIMATE.noise_model, "enable_adapters")
if not _HAS_ADAPTER_TOGGLE:
    print("[boot] WARNING: noise_model has no enable_adapters()/disable_adapters(); the relighting "
          "LoRA will stay active for 'animate' jobs too (see README's known-uncertainty note)",
          flush=True)
print(f"[boot] WanAnimate ready in {time.time() - _t0:.1f}s (t5_cpu={T5_CPU}, "
      f"adapter_toggle={_HAS_ADAPTER_TOGGLE})", flush=True)

# process_checkpoint/det/yolov10m.onnx is never fetched (AGPL-3.0; see README.md) -- ProcessPipeline's
# constructor only requires det_checkpoint_path to be non-None to build a detector at all;
# detector_adapter.PersonDetector ignores the value entirely.
_POSE2D_CKPT = os.path.join(WAN_ANIMATE_DIR, "process_checkpoint", "pose2d", "vitpose_h_wholebody.onnx")
_SAM2_CKPT = os.path.join(WAN_ANIMATE_DIR, "process_checkpoint", "sam2", "sam2_hiera_large.pt")

print("[boot] loading preprocessing pipeline (ViTPose + person detector + SAM2)...", flush=True)
PIPELINE = ProcessPipeline(
    det_checkpoint_path="replaced-by-detector_adapter.PersonDetector",
    pose2d_checkpoint_path=_POSE2D_CKPT,
    sam_checkpoint_path=_SAM2_CKPT,  # built once; only exercised by "replace" jobs
    flux_kontext_path=None,         # FLUX-enhanced retargeting is out of scope here; see README
)
print(f"[boot] ready in {time.time() - _t0:.1f}s total", flush=True)


def _fetch(src: str, dst: str) -> None:
    """src is a data: URL or an https URL (job inputs arrive as either)."""
    if src.startswith("data:"):
        _, b64 = src.split(",", 1)
        with open(dst, "wb") as f:
            f.write(base64.b64decode(b64, validate=True))
        return
    with urllib.request.urlopen(src, timeout=120) as r, open(dst, "wb") as f:
        f.write(r.read())


def _probe(path: str) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        return json.loads(out.stdout or "{}")
    except json.JSONDecodeError:
        return {}


def _parse_resolution(value) -> list:
    """'1280x720' -> [1280, 720]. Only the PRODUCT reaches the official preprocessing
    (wan/modules/animate/preprocess/utils.py:resize_by_area -> calculate_new_size derives the actual
    output width/height from the SOURCE frame's own aspect ratio and this area, confirmed by reading
    both functions 2026-10-17) -- so [1280,720] and [720,1280] are equivalent inputs. Accepting
    either order is just for a natural-feeling API, not because the order changes anything."""
    if not value:
        return DEFAULT_RESOLUTION_AREA
    parts = str(value).lower().replace("*", "x").split("x")
    return [int(parts[0]), int(parts[1])]


def _tensor_to_mp4(tensor: torch.Tensor, fps: float, out_path: str) -> None:
    """tensor: (C, N, H, W) float, values in [-1, 1] -- wan.WanAnimate.generate()'s own output range
    and layout (its docstring says so explicitly). Reproduces wan.utils.utils.save_video's exact
    normalization (clamp to value_range, (x+1)*127.5 == torchvision.utils.make_grid's
    normalize=True, value_range=(-1,1) math for a single-item batch) but pipes frames straight to
    ffmpeg instead of through imageio, for one encode instead of two and to guarantee this handler's
    yuv420p + faststart contract on the exact file it returns, matching vid-wan-a14b/handler.py's
    _frames_to_mp4 pattern."""
    frames = tensor.clamp(-1, 1).add(1).mul(127.5).round().clamp(0, 255).to(torch.uint8)
    frames = frames.permute(1, 2, 3, 0).contiguous().cpu().numpy()  # N, H, W, C
    n, h, w = frames.shape[0], frames.shape[1], frames.shape[2]
    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
        "-r", str(fps), "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "medium", "-crf", "17",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path,
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    for i in range(n):
        p.stdin.write(frames[i].tobytes())
    p.stdin.close()
    err = p.stderr.read().decode(errors="ignore")[-800:]
    if p.wait() != 0:
        raise RuntimeError(f"ffmpeg encode failed: {err}")


def _finalize(raw_video_path: str, audio_source_path, out_path: str) -> None:
    """raw_video_path is already h264/yuv420p/+faststart (see _tensor_to_mp4). Mux the driving
    video's own audio back in when it has one (ffmpeg, aac, shortest -- same shape as Wan2.2's own
    wan.utils.utils.merge_video_audio, plus +faststart, which that function does not set)."""
    if audio_source_path is None:
        os.replace(raw_video_path, out_path)
        return
    cmd = [
        "ffmpeg", "-y", "-i", raw_video_path, "-i", audio_source_path,
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v:0", "-map", "1:a:0", "-shortest",
        "-movflags", "+faststart", out_path,
    ]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg audio mux failed: {r.stderr[-800:]}")


def handler(job):
    job_input = job.get("input") or {}
    image_src = job_input.get("image")
    video_src = job_input.get("video")
    if not image_src:
        return {"error": "image is required (the character)"}
    if not video_src:
        return {"error": "video is required (the driving performance)"}

    mode = str(job_input.get("mode") or "animate").lower()
    if mode not in ("animate", "replace"):
        return {"error": f"mode must be 'animate' or 'replace', got {mode!r}"}
    replace_flag = mode == "replace"

    prompt = job_input.get("prompt") or CFG.prompt
    seed = job_input.get("seed")
    seed = int(seed) if seed is not None and int(seed) >= 0 else int.from_bytes(os.urandom(4), "big")
    steps = int(job_input.get("steps") or CFG.sample_steps)
    max_seconds = float(job_input.get("max_seconds", 10) or 10)
    trim = bool(job_input.get("trim", False))
    resolution_area = _parse_resolution(job_input.get("resolution"))

    started = time.time()
    work = tempfile.mkdtemp(prefix="wananimate_")
    try:
        image_path = os.path.join(work, "character.png")
        video_path = os.path.join(work, "driving.mp4")
        _fetch(image_src, image_path)
        _fetch(video_src, video_path)

        info = _probe(video_path)
        fmt = info.get("format", {})
        duration = float(fmt.get("duration", 0) or 0)
        has_audio = any(s.get("codec_type") == "audio" for s in info.get("streams", []))

        if duration > max_seconds:
            if not trim:
                return {"error": f"driving video is {duration:.1f}s, over max_seconds={max_seconds}. "
                                 "Pass trim: true to use the first max_seconds instead."}
            trimmed_path = os.path.join(work, "driving_trimmed.mp4")
            r = subprocess.run(["ffmpeg", "-y", "-i", video_path, "-t", str(max_seconds),
                               "-c", "copy", trimmed_path],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if r.returncode != 0:
                return {"error": f"ffmpeg trim failed: {r.stderr[-800:]}"}
            video_path = trimmed_path

        # Toggle the relighting LoRA per job (see module docstring): the official code only ever
        # loads it for a whole replace-mode process, never alongside animate-mode in the same one.
        if _HAS_ADAPTER_TOGGLE:
            if replace_flag:
                ANIMATE.noise_model.enable_adapters()
            else:
                ANIMATE.noise_model.disable_adapters()

        pre_t0 = time.time()
        PIPELINE(
            video_path=video_path,
            refer_image_path=image_path,
            output_path=work,
            resolution_area=resolution_area,
            fps=CFG.sample_fps,              # the model's native rate (30); see _parse_resolution's
                                              # neighbour note -- generate.py always encodes output at
                                              # this same fixed rate, so preprocessing must match it.
            replace_flag=replace_flag,
            retarget_flag=not replace_flag,  # UserGuider.md: recommended ON for animate, OFF for
                                              # replace (deliberate upstream default, not a guess).
            use_flux=False,
            iterations=3, k=7, w_len=1, h_len=1,
        )
        print(f"[run] preprocessing done in {time.time() - pre_t0:.1f}s (mode={mode})", flush=True)

        gen_t0 = time.time()
        out_tensor = ANIMATE.generate(
            src_root_path=work,
            replace_flag=replace_flag,
            clip_len=CFG.frame_num,
            refert_num=5,  # "recommended to be 1 or 5" (docstring); 5 for better cross-chunk
                           # continuity at a few extra frames of overlap, not generate.py's own CLI
                           # default of 77 -- that default fails WanAnimate.generate()'s own
                           # `assert refert_num == 1 or refert_num == 5` and looks like a copy/paste
                           # bug from --frame_num, not something to propagate here.
            shift=CFG.sample_shift,
            sample_solver="dpm++",
            sampling_steps=steps,
            guide_scale=float(job_input.get("guide_scale", CFG.sample_guide_scale) or CFG.sample_guide_scale),
            input_prompt=prompt,
            seed=seed,
            offload_model=True,
        )
        gen_seconds = time.time() - gen_t0
        print(f"[run] generation done in {gen_seconds:.1f}s", flush=True)

        _, frames, height, width = out_tensor.shape
        raw_path = os.path.join(work, "raw.mp4")
        _tensor_to_mp4(out_tensor, CFG.sample_fps, raw_path)

        final_path = os.path.join(work, "final.mp4")
        _finalize(raw_path, video_path if has_audio else None, final_path)

        with open(final_path, "rb") as f:
            video_bytes = f.read()

        return {
            "video_url": "data:video/mp4;base64," + base64.b64encode(video_bytes).decode(),
            "duration_seconds": round(frames / CFG.sample_fps, 2),
            "width": int(width),
            "height": int(height),
            "fps": float(CFG.sample_fps),
            "frames": int(frames),
            "mode": mode,
            "generation_seconds": round(gen_seconds, 2),
        }
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        return {"error": f"CUDA OOM: {e}. Shorten the driving clip (max_seconds) or lower resolution."}
    except Exception as e:  # surface the reason rather than dying silently
        return {"error": f"{type(e).__name__}: {e}"}
    finally:
        shutil.rmtree(work, ignore_errors=True)
        print(f"[run] job finished in {time.time() - started:.1f}s total", flush=True)


import runpod  # noqa: E402

runpod.serverless.start({"handler": handler})
