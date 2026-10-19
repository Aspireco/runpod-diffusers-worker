"""Run at image build time: fail the BUILD, not a billed GPU, when the Wan-Animate stack is wrong.

    python /build_check.py imports     every package the handler imports actually imports
    python /build_check.py wan         wan.WanAnimate and the animate-14B config resolve
    python /build_check.py detector    the AGPL YOLOv10 detector is gone and the replacement is wired
    python /build_check.py sam2        sam2's CUDA extension built (process_checkpoint/sam2, replace mode)
    python /build_check.py sam2cfg     Wan's SAM2 predictor build finds sam2_hiera_l.yaml and constructs, on CPU

A file rather than RUN python -c "..." continuation lines: ACR's dependency scanner reads any line that
starts with `from` as a FROM instruction and fails the build before it begins (vid-wan-a14b build_check.py,
2026-10-17). This file imports everything handler.py imports, so a broken dependency fails here, at build
time, in a few seconds of CPU, rather than on a billed A100 minutes into a job.
"""
import inspect
import sys

import torch


def _patch_cuda_current_device_for_cpu_build() -> None:
    """wan/modules/t5.py:478 writes `device=torch.cuda.current_device()` as a DEFAULT ARGUMENT
    VALUE on T5EncoderModel.__init__ -- Python evaluates that at class-definition time, i.e. the
    moment `import wan` reaches `wan/modules/__init__.py`'s `from .t5 import ...`, not lazily when
    the class is later instantiated. ACR's build agents have no NVIDIA driver at all (found live,
    2026-10-17, ACR run `ch1x`: `RuntimeError: Found no NVIDIA driver on your system`), so a plain
    `import wan` cannot complete there -- not something a GPU-less build box can ever satisfy, since
    `torch.cuda.current_device()` tries to initialize a CUDA context. Patched here, for this
    CPU-only build-time check ONLY: the real worker runs on an actual A100 where this call succeeds
    normally, so handler.py needs no equivalent patch. Harmless because nothing in these checks
    instantiates T5EncoderModel (that needs the real 11GB checkpoint); only the class definition
    itself must survive being imported."""
    if not torch.cuda.is_available():
        torch.cuda.current_device = lambda: 0


def imports() -> None:
    # The three packages Wan2.2's own requirements.txt / requirements_animate.txt do NOT list, despite
    # unconditional top-level imports in wan/animate.py, wan/modules/animate/*.py and
    # wan/modules/animate/preprocess/*.py: einops (wan/animate.py, model_animate.py, face_blocks.py,
    # motioner.py, auxi_blocks.py, s2v/*.py -- all pulled in eagerly by `import wan`), moviepy
    # (process_pipepline.py writes src_pose.mp4/src_face.mp4/src_bg.mp4/src_mask.mp4 with it), and
    # librosa (wan/modules/s2v/audio_encoder.py, eagerly imported via wan/__init__.py's `from
    # .speech2video import WanS2V` even though this worker never runs speech2video).
    import einops  # noqa: F401
    import librosa  # noqa: F401
    import moviepy  # noqa: F401

    # onnxruntime must be the GPU build: requirements_animate.txt lists plain "onnxruntime" (CPU-only),
    # but pose2d.py's SimpleOnnxInference asks for CUDAExecutionProvider. The plain package doesn't
    # compile that provider in at all, so ViTPose would silently run on CPU for every driving-video
    # frame instead of raising -- a slow, quiet bug, not a visible one. Install onnxruntime-gpu instead.
    import onnxruntime
    providers = onnxruntime.get_available_providers()
    assert "CUDAExecutionProvider" in providers, (
        f"onnxruntime has no CUDAExecutionProvider ({providers}); onnxruntime-gpu did not install "
        "correctly -- ViTPose pose extraction would silently run on CPU"
    )
    print("onnxruntime", onnxruntime.__version__, "providers:", providers)

    import cv2  # noqa: F401
    import decord  # noqa: F401
    import diffusers  # noqa: F401
    import ftfy  # noqa: F401
    import peft  # noqa: F401
    import sam2  # noqa: F401
    import transformers  # noqa: F401
    from PIL import Image  # noqa: F401
    from torchvision.models.detection import (  # noqa: F401
        FasterRCNN_ResNet50_FPN_V2_Weights,
        fasterrcnn_resnet50_fpn_v2,
    )
    print("imports OK: einops, librosa, moviepy, onnxruntime(-gpu), cv2, decord, diffusers, ftfy, "
          "peft, sam2, transformers, torchvision.models.detection")


def wan_() -> None:  # trailing underscore: the function body does `import wan`, same reason sam2_ has one
    _patch_cuda_current_device_for_cpu_build()
    import wan
    from wan.configs import WAN_CONFIGS
    assert hasattr(wan, "WanAnimate"), "wan package has no WanAnimate -- wrong commit or broken install"
    assert "animate-14B" in WAN_CONFIGS, f"no animate-14B config: {sorted(WAN_CONFIGS)}"
    cfg = WAN_CONFIGS["animate-14B"]
    for key in ("t5_checkpoint", "clip_checkpoint", "vae_checkpoint", "lora_checkpoint",
                "t5_tokenizer", "clip_tokenizer"):
        assert key in cfg, f"animate-14B config missing {key}"
    sig = set(inspect.signature(wan.WanAnimate.__init__).parameters)
    assert "use_relighting_lora" in sig, f"wan.WanAnimate lost use_relighting_lora: {sorted(sig)}"
    gen_sig = set(inspect.signature(wan.WanAnimate.generate).parameters)
    for p in ("src_root_path", "replace_flag", "clip_len", "refert_num"):
        assert p in gen_sig, f"wan.WanAnimate.generate lost {p}: {sorted(gen_sig)}"

    # handler.py serves both "animate" and "replace" jobs from ONE resident model (loading a second
    # 14B model just to flip one constructor flag would roughly double VRAM for no benefit), toggling
    # the relighting LoRA per job via diffusers' PeftAdapterMixin instead of generate.py's one-model-
    # one-mode-per-process assumption. That needs these two methods on the noise model class.
    from wan.modules.animate import WanAnimateModel
    assert hasattr(WanAnimateModel, "enable_adapters") and hasattr(WanAnimateModel, "disable_adapters"), (
        "WanAnimateModel has no enable_adapters()/disable_adapters() -- handler.py's per-job "
        "relighting-LoRA toggle will not work; see its module docstring"
    )
    print("wan", getattr(wan, "__version__", "?"), "| WanAnimate OK, animate-14B config OK "
          f"(t5={cfg.t5_checkpoint}, clip={cfg.clip_checkpoint}, vae={cfg.vae_checkpoint}, "
          f"lora={cfg.lora_checkpoint}), WanAnimateModel has enable/disable_adapters")


def detector() -> None:
    """Our torchvision replacement must present the exact interface pose2d.Pose2d calls, so the
    monkey-patch in handler.py (`pose2d.Yolo = detector_adapter.PersonDetector`) is a true drop-in.
    The AGPL yolov10m.onnx itself is never in the image to begin with -- nothing in this Dockerfile
    downloads it, and gpu/manifests/wan-animate*.tsv excludes process_checkpoint/det/ -- so there is
    no file for a build-time check to look for; this checks the code path that replaces it instead."""
    _patch_cuda_current_device_for_cpu_build()
    import wan
    import os
    preprocess_dir = os.path.join(os.path.dirname(wan.__file__), "modules", "animate", "preprocess")
    assert os.path.isdir(preprocess_dir), f"no preprocess dir at {preprocess_dir}"

    sys.path.insert(0, preprocess_dir)
    import pose2d  # the vendored module we monkey-patch, not our own

    import detector_adapter
    pose2d.Yolo = detector_adapter.PersonDetector  # the actual patch handler.py applies at boot

    adapter = detector_adapter.PersonDetector(checkpoint="unused", device="cpu")
    import numpy as np
    frame = (np.random.rand(64, 48, 3) * 255).astype("uint8")
    img, shape = adapter.preprocess(frame)
    assert img.shape == (3, 64, 48), f"unexpected preprocess shape {img.shape}"
    result = adapter(img[None], shape[None])
    assert len(result) == 1 and len(result[0]) == 1 and "bbox" in result[0][0], (
        f"PersonDetector broke the Pose2d contract: {result}")
    assert len(result[0][0]["bbox"]) == 5, "bbox must be [x1, y1, x2, y2, score]"
    print("detector OK: pose2d.Yolo patched to detector_adapter.PersonDetector "
          "(torchvision Faster R-CNN v2, no yolov10m.onnx anywhere)")


def sam2_() -> None:
    import sam2
    from sam2 import _C  # the compiled CUDA extension connected_components.cu builds into
    from sam2.build_sam import _load_checkpoint  # noqa: F401
    from sam2.utils.misc import fill_holes_in_mask_scores  # noqa: F401
    assert hasattr(_C, "get_connected_componnets"), "sam2._C built without get_connected_componnets"
    print("sam2", getattr(sam2, "__version__", "?"), "| CUDA extension sam2._C built "
          "(needed: Wan2.2's build_sam2_video_predictor sets fill_hole_area=8 by default)")


def sam2cfg() -> None:
    """Wan2.2's own SAM2 predictor build (preprocess/sam_utils.py build_sam2_video_predictor, the call
    ProcessPipeline makes at boot), on CPU: hydra must find sam2_hiera_l.yaml through the search path
    `import sam2` registers (pkg://sam2_configs), and every _target_ in it must construct. sam2 at the
    pinned commit keeps those yaml files as data that neither its setup.py (no package_data) nor a
    MANIFEST.in ships, so the non-editable install left sam2_configs/ holding only __init__.py and the
    first Azure replace proof died at boot with MissingConfigException (2026-10-18, job
    2026-10-18.8207657749-ed285b93, after a 19-minute cold start). Construction never touches CUDA
    (only init_state and forward do). Optional argv[2]: a sam2_hiera_large.pt to load as well, which
    the pre-flight smoke run downloads from Wan's own repo; the image build has none."""
    import os
    _patch_cuda_current_device_for_cpu_build()
    import wan
    sys.path.insert(0, os.path.join(os.path.dirname(wan.__file__), "modules", "animate", "preprocess"))
    import sam2  # noqa: F401  (registers hydra's search path)
    import sam2_configs
    cfg_dir = os.path.dirname(sam2_configs.__file__)
    yamls = sorted(f for f in os.listdir(cfg_dir) if f.endswith(".yaml"))
    assert "sam2_hiera_l.yaml" in yamls, f"no sam2_hiera_l.yaml in {cfg_dir}: {sorted(os.listdir(cfg_dir))}"
    from sam_utils import build_sam2_video_predictor  # Wan's flat module, as process_pipepline.py imports it
    ckpt = sys.argv[2] if len(sys.argv) > 2 else None
    predictor = build_sam2_video_predictor("sam2_hiera_l.yaml", ckpt, device="cpu")
    params = sum(p.numel() for p in predictor.parameters()) / 1e6
    print(f"sam2 configs OK in {cfg_dir} {yamls}; Wan's build_sam2_video_predictor built "
          f"{type(predictor).__module__}.{type(predictor).__name__} on CPU ({params:.0f}M parameters), "
          f"checkpoint {'loaded with no missing keys: ' + ckpt if ckpt else 'not loaded (none in the image)'}")


if __name__ == "__main__":
    assert "cu13" not in torch.__version__, f"torch replaced: {torch.__version__}"
    {"imports": imports, "wan": wan_, "detector": detector, "sam2": sam2_, "sam2cfg": sam2cfg}[sys.argv[1]]()
