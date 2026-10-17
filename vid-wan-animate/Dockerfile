# Wan2.2-Animate-14B (Apache-2.0): a character image + a driving video -> the character performs
# the video's motion ("animate") or replaces the person inside it ("replace"). Weights come from the
# Azure Files share at $WAN_ANIMATE_DIR (default /runpod-volume/wan-animate), staged by the fetch
# job from gpu/manifests/wan-animate.tsv + wan-animate-replace.tsv. The image itself holds none of
# them except a ~170MB detector checkpoint (see the torchvision RUN step below). Full licence table,
# the AGPL-detector fix and the "official code paths" reasoning are in README.md.
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV HF_HUB_ENABLE_HF_TRANSFER=0 \
    HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    WAN_ANIMATE_DIR="/runpod-volume/wan-animate" \
    TORCH_CUDA_ARCH_LIST="8.0" \
    PYTHONUNBUFFERED=1

# ffmpeg: the handler's own encode/mux step (frames -> h264/yuv420p/+faststart, then mux the driving
# video's audio back in) and ffprobe (duration/fps/audio-stream detection). libsndfile1: librosa's
# audio backend -- needed only because `import wan` eagerly imports wan.speech2video, which eagerly
# imports wan.modules.s2v.audio_encoder, which eagerly imports librosa; this lane never calls
# speech2video, but the chain still has to import cleanly.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --ignore-installed cryptography
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Pin the base image's torch as constraints (see vid-wan-a14b/Dockerfile for the full story): left
# unpinned, diffusers/accelerate/peft's bare "torch" dependency resolves to a CUDA 13 wheel from
# PyPI, which needs driver >= 580 and crash-loops with an empty log before Python even starts.
RUN python -c "import torch, torchvision, torchaudio; \
open('/tmp/constraints.txt','w').write(''.join(f'{n}=={v}\n' for n,v in \
[('torch',torch.__version__),('torchvision',torchvision.__version__),('torchaudio',torchaudio.__version__)]))" \
    && cat /tmp/constraints.txt

# flash-attn is a HARD runtime dependency here, not an optional accelerator: wan/modules/animate/
# clip.py and model_animate.py call flash_attn.flash_attn_varlen_func directly (confirmed by reading
# both files, 2026-10-17) rather than going through wan/modules/attention.py's attention() dispatcher,
# which is the one with a torch-SDPA fallback. Compiling flash-attn from source on ACR takes hours
# (many template-heavy CUDA kernels); a prebuilt wheel takes seconds. The exact build this base image
# needs -- Python 3.12 (cp312), torch 2.8, CUDA 12, cxx11 ABI TRUE -- was confirmed by probing the
# base image directly on 2026-10-17 (`python --version` -> 3.12.3, `torch.version.cuda` -> 12.8,
# `torch.compiled_with_cxx11_abi()` -> True), not assumed: ACR run `ch1p`, registry azswmcacr,
# wan-animate-probe:tmp (throwaway, not part of this lane).
RUN pip install --no-cache-dir -c /tmp/constraints.txt \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post2/flash_attn-2.8.3.post2+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

# requirements.txt + requirements_animate.txt from the Wan2.2 repo (see below for the pinned commit),
# combined, minus flash_attn (already installed above from a wheel), PLUS three packages that file
# set is missing despite unconditional top-level imports in code it covers -- einops (wan/animate.py,
# model_animate.py, face_blocks.py, motioner.py, auxi_blocks.py, s2v/*.py), moviepy
# (preprocess/process_pipepline.py writes src_pose.mp4 etc. with it) and librosa
# (modules/s2v/audio_encoder.py, pulled in eagerly by `import wan` even though this lane never
# calls speech2video) -- found by grepping every file `import wan` actually touches at module load
# time, not just the two requirements files, 2026-10-17. onnxruntime-gpu replaces plain
# "onnxruntime": pose2d.py's SimpleOnnxInference explicitly asks for CUDAExecutionProvider, which the
# CPU-only PyPI package never compiles in at all -- ViTPose would silently run on CPU every frame
# instead of raising, a slow bug rather than a loud one.
RUN pip install --no-cache-dir -c /tmp/constraints.txt \
      "diffusers>=0.31.0" \
      "transformers>=4.49.0,<=4.51.3" \
      "tokenizers>=0.20.3" \
      "accelerate>=1.1.1" \
      "opencv-python>=4.9.0.80" \
      "numpy>=1.23.5,<2" \
      "easydict" "ftfy" "dashscope" "imageio" "imageio-ffmpeg" "tqdm" \
      "decord" "peft" "onnxruntime-gpu" "pandas" "matplotlib" "loguru" "sentencepiece" \
      "einops" "moviepy" "librosa" \
      "runpod>=1.10.1"

# Wan2.2 itself, Apache-2.0, pinned to the commit this worker was built and read against
# (github.com/Wan-Video/Wan2.2, 1ea34ff48f87168174e12956e200b1d908b1c5ff, 2026-09-21). Its own
# pyproject.toml declares `packages = ["wan"]` plus a package-data glob of `**/*.py`, which ships
# the FULL wan/ tree -- including modules/animate/preprocess/ -- as installed files even though
# setuptools would not otherwise treat it as a subpackage. handler.py finds that directory at
# runtime via os.path.dirname(wan.__file__) and adds it to sys.path, because preprocess_data.py,
# process_pipepline.py and friends use flat, non-relative imports ("from pose2d import Pose2d") and
# were written to run as scripts from inside that directory, not as an importable subpackage --
# confirmed by reading preprocess/__init__.py, which itself does a RELATIVE import that only works
# if nothing in `import wan` ever reaches it (it doesn't: checked wan/__init__.py, modules/__init__.py
# and modules/animate/__init__.py directly).
RUN pip install --no-cache-dir -c /tmp/constraints.txt \
    "wan @ git+https://github.com/Wan-Video/Wan2.2.git@1ea34ff48f87168174e12956e200b1d908b1c5ff"

# SAM 2 (Apache-2.0, Meta), pinned to the exact commit Wan2.2's OWN requirements_animate.txt
# specifies -- "use the official code paths" extends to the pinned dependency, not just Wan2.2
# itself. Its setup.py always compiles a small CUDA extension (csrc/connected_components.cu) with no
# env-var skip at this commit, and it is NOT dead code here: wan/modules/animate/preprocess/
# sam_utils.py's build_sam2_video_predictor() sets `++model.fill_hole_area=8` by default, which calls
# the compiled op at runtime in "replace" mode. TORCH_CUDA_ARCH_LIST=8.0 (set above) targets only the
# card this runs on (A100, sm_80) so the compile is one architecture, not every one torch shipped
# with. --no-build-isolation: setup.py does `from torch.utils.cpp_extension import ...` at build
# time, which needs torch visible in the SAME environment, not PEP 517's isolated build sandbox --
# the identical gotcha flash-attn's own INSTALL.md documents for itself.
RUN pip install --no-cache-dir --no-build-isolation -c /tmp/constraints.txt \
    "git+https://github.com/facebookresearch/sam2.git@0e78a118995e66bb27d78518c4bd9a3e95b4e266"

# The ONE licence fix this worker exists to make: Wan-Animate's official preprocessing detector
# (process_checkpoint/det/yolov10m.onnx, Ultralytics YOLOv10) is AGPL-3.0 and is never downloaded --
# see README.md's licence table. detector_adapter.py replaces it with torchvision's own Faster R-CNN
# v2, COCO-pretrained (BSD-3-Clause code and weights, same licence as torchvision itself). Baking the
# ~170MB checkpoint into the image here -- well under the "small aux files" allowance -- means no
# first-request download and no runtime dependency on download.pytorch.org being reachable from
# Sweden Central.
RUN python -c "from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights as W; \
fasterrcnn_resnet50_fpn_v2(weights=W.COCO_V1)"

# Fail the BUILD, not a billed GPU. build_check.py imports everything handler.py imports.
COPY build_check.py /build_check.py
COPY detector_adapter.py /detector_adapter.py
RUN python -c "import torch; v=torch.__version__; print('torch', v); assert 'cu13' not in v, f'torch replaced: {v}'"
RUN python /build_check.py imports
RUN python /build_check.py wan
RUN python /build_check.py detector
RUN python /build_check.py sam2

COPY handler.py /handler.py
RUN python -c "import ast; ast.parse(open('/handler.py').read()); print('handler parses OK')"

CMD ["python", "-u", "/handler.py"]
