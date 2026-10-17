# MuseTalk 1.5 lip-sync (TMElyralab/MuseTalk, MIT) for the Azure GPU fleet lane `musetalk` (T4, 16 GB).
# No weights baked in except the one small S3FD face-detector checkpoint (~86 MB, see below); the rest
# live on the Azure Files share at $MUSETALK_DIR, staged by the fetch job from the manifests next to this
# file. See README.md for the full contract, licence table and premade-search notes.
#
# Base pinned to torch 2.0.1 / CUDA 11.7 because MuseTalk's stack (mmcv 2.0.1, mmdet 3.1.0, mmpose 1.1.0)
# only ships prebuilt wheels for that exact torch/CUDA combo (openmmlab's own wheel index); torch is never
# reinstalled from default PyPI for the same reason every other lane in this repo avoids it (newer wheels
# want a newer driver than the mmlab wheels were built against). CUDA 11.7 containers run fine under the
# T4 host's driver 580 (a newer driver runs older-CUDA containers forward-compatibly).
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    MUSETALK_DIR=/runpod-volume/musetalk \
    DTYPE=float16 \
    DEBIAN_FRONTEND=noninteractive

# ffmpeg: frame extraction + final mux. libgl1/libglib2.0-0: opencv-python's usual
# "libGL.so.1: cannot open shared object file" headless-container gotcha. build-essential:
# mmpose's xtcocotools (and a couple of its own transitive deps) build a C/Cython extension
# at pip-install time; the base image's "devel" CUDA layer does not itself guarantee a host
# gcc. git: pinned clone of the MuseTalk repo itself. curl: the one baked-in aux weight below.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg git curl build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# cryptography arrives via apt with no pip record on some base images; shadow instead of
# fighting pip's "Cannot uninstall ... no-record-file" when runpod's chain wants to upgrade it.
RUN pip install --no-cache-dir --ignore-installed cryptography
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Pinned to the commit current at build time (2026-09-26), not floating `main`.
RUN git clone https://github.com/TMElyralab/MuseTalk.git /opt/MuseTalk \
    && cd /opt/MuseTalk && git checkout 0a89dec45a0192b824e3cf4daf96c239440c5ed8 \
    && rm -rf .git

WORKDIR /opt/MuseTalk

# MuseTalk's own requirements.txt, pinned versions exactly as published, minus the packages this
# handler's call path never imports (traced import-by-import through scripts/inference.py and
# every musetalk.utils.* module it uses): tensorflow==2.12.0/tensorboard==2.12.0 (a
# training/eval-only metric, not touched by scripts/inference.py), gradio==5.24.0 (app.py's demo
# UI only), and the weight-download helpers gdown/requests plus the alternate video I/O libs
# imageio[ffmpeg]/ffmpeg-python/moviepy (scripts/inference.py shells out to the ffmpeg binary
# directly via subprocess instead). Dropping all eight keeps a meaningfully leaner image; see
# README.md "Premade search" for the reasoning. accelerate is kept: diffusers' from_pretrained
# can reach for it internally even on a plain load.
RUN pip install --no-cache-dir \
      diffusers==0.30.2 accelerate==0.28.0 numpy==1.23.5 opencv-python==4.9.0.80 \
      soundfile==0.12.1 transformers==4.39.2 "huggingface_hub==0.30.2" librosa==0.11.0 \
      einops==0.8.1 omegaconf

# MMLab stack, exactly as MuseTalk's README installs it. The wheel index pins mmcv to a
# prebuilt binary for this torch/CUDA pair instead of letting mim/pip try to build it from
# source (and possibly re-resolve torch in the process).
#
# chumpy==0.70 (an mmpose transitive dependency, for an SMPL body-model utility this handler's
# call path never reaches) has to be installed separately first, with --no-build-isolation: its
# setup.py does a bare `import pip` at module scope (a long-dead pre-PEP517 pattern -- chumpy's
# last release was 2017), which only resolves when setup.py runs in the ambient environment
# (where the pip package is genuinely importable) instead of pip's normal isolated build venv
# (which has no pip package inside it at all, by design). Confirmed against a real failing build
# on 2026-10-17: "ModuleNotFoundError: No module named 'pip'" from inside chumpy's own
# get_requires_for_build_wheel hook. numpy/scipy/six (chumpy's own setup-time imports) are
# already installed by the pip layer above, which --no-build-isolation requires.
RUN pip install --no-cache-dir -U openmim \
    && pip install --no-cache-dir mmengine \
    && pip install --no-cache-dir mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0.0/index.html \
    && pip install --no-cache-dir --no-build-isolation "chumpy==0.70" \
    && pip install --no-cache-dir mmdet==3.1.0 mmpose==1.1.0

RUN pip install --no-cache-dir "runpod>=1.10.1"

# S3FD face detector weight (BSD-3-Clause, Adrian Bulat 2017 -- vendored into MuseTalk's own
# musetalk/utils/face_detection/detection/sfd/). Upstream's own sfd_detector.py auto-downloads
# this from adrianbulat.com at first use if the file is absent; baking it into the image at
# this exact path (SFDDetector's own default: alongside sfd_detector.py itself) means every
# cold start needs zero 3rd-party network calls. 86 MB -- well under the "tiny aux file"
# allowance, so it travels in the image rather than through the fetch-job/manifest pipeline.
RUN curl -fL --retry 3 --retry-delay 5 -o musetalk/utils/face_detection/detection/sfd/s3fd.pth \
      https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth \
    && test "$(wc -c < musetalk/utils/face_detection/detection/sfd/s3fd.pth)" = "89843225"

COPY build_check.py /build_check.py
RUN python /build_check.py

COPY handler.py /handler.py
RUN python -c "import ast; ast.parse(open('/handler.py').read()); print('handler parses OK')"

CMD ["python", "-u", "/handler.py"]
