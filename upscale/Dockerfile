# Real-ESRGAN upscaling + RIFE frame interpolation for the Azure GPU fleet lanes `upscale`
# and `interpolate` (T4, 16GB). Both sets of weights are baked in: ~160MB total, small
# enough that baking beats a first-boot download and removes the runtime network dependency
# entirely (on RunPod this also removes the need for its separate Model Caching setup).
#
# One image serves both jobs -- deploy it twice, MODE=upscale and MODE=interpolate.
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# The torch280 base ships HF_HUB_ENABLE_HF_TRANSFER=1 without the hf_transfer package,
# so the first runtime download from HuggingFace dies with "Fast download using
# 'hf_transfer' is enabled ... but 'hf_transfer' package is not available". Found live on
# the music lane 2026-09-14; every lane on this base inherits it. Turning the fast path
# off costs nothing -- it was never installed to begin with.
ENV HF_HUB_ENABLE_HF_TRANSFER=0

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf \
    HF_HUB_DISABLE_TELEMETRY=1 \
    MODE=upscale \
    WEIGHTS_DIR=/opt/weights \
    RIFE_DIR=/opt/rife

# ffmpeg does all video I/O -- frames stream through it as raw RGB in both directions, so
# nothing is ever written to disk as a PNG sequence. git/unzip/curl are build-time only.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg curl git unzip \
    && rm -rf /var/lib/apt/lists/*

# cryptography ships via apt in the base image with no pip record, so pip cannot uninstall
# it for runpod's dependency chain. Shadow it instead of trying to remove it.
RUN pip install --no-cache-dir --ignore-installed cryptography

# Python 3.12 dropped pkgutil.ImpImporter; legacy setup.py dependencies fail without this.
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# spandrel loads Real-ESRGAN .pth files directly, detecting RRDBNet and unwrapping the
# params_ema key on its own. It replaces the realesrgan/basicsr pair, which is unusable
# here: basicsr imports torchvision.transforms.functional_tensor, removed in torchvision
# 0.17, and this base image ships 0.23.
#
# --no-deps is deliberate. spandrel's metadata requires bare "torch" and "torchvision";
# an unconstrained requirement should leave an installed build alone, but the cost of
# being wrong is CUDA 13 wheels, a driver mismatch and a crash with no log line. Its four
# real dependencies are listed by hand instead. Pinned exactly because --no-deps means a
# new release could add a dependency we would not install.
RUN pip install --no-cache-dir --no-deps "spandrel==0.4.2"

RUN pip install --no-cache-dir \
      "safetensors" \
      "einops" \
      "typing_extensions" \
      "transformers>=4.50.3,<5" \
      "huggingface-hub<1.0" \
      "runpod>=1.10.1"

# Real-ESRGAN weights from xinntao's own releases (BSD-3-Clause). Both scales are baked so
# the handler can switch on `scale` at import time -- a replica keeps calling the handler
# until its job queue is empty before it exits (gpu/azure_runner.py's run loop), so anything
# loaded inside the handler function body would be re-paid on every job, not every cold start.
RUN mkdir -p /opt/weights \
 && curl -fsSL -o /opt/weights/RealESRGAN_x4plus.pth \
      https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
 && curl -fsSL -o /opt/weights/RealESRGAN_x2plus.pth \
      https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth

# RIFE: the architecture lives in the repo (MIT) but the weights ship as a separate
# archive. The README points at Google Drive, whose virus-scan interstitial cannot be
# scripted; hzwer's own HuggingFace mirror carries the identical v4.26 release.
#
# requirements.txt is deliberately NOT installed: it pins numpy<=1.23.5, which has no
# cp312 wheel and would try to build from source, and pulls sk-video/moviepy that this
# handler replaces with direct ffmpeg calls. The model needs nothing but torch.
RUN git clone --depth 1 https://github.com/hzwer/Practical-RIFE.git /opt/rife \
 && curl -fsSL -o /tmp/rife.zip \
      https://huggingface.co/hzwer/RIFE/resolve/main/RIFEv4.26_0921.zip \
 && unzip -q /tmp/rife.zip -d /tmp/rife \
 && mkdir -p /opt/rife/train_log \
 && cp /tmp/rife/RIFEv4.26_0921/flownet.pkl /tmp/rife/RIFEv4.26_0921/*.py /opt/rife/train_log/ \
 && rm -rf /tmp/rife /tmp/rife.zip /opt/rife/.git /opt/rife/demo

COPY handler.py /handler.py

# Fail the BUILD rather than the worker: a syntax error surfaces here, where the error is
# visible, instead of as a silent crash-loop with no container logs.
RUN python -c "import ast; ast.parse(open('/handler.py').read()); print('handler parses OK')"

# Fail the BUILD, not a billed GPU: runs handler.py's own module-level construction for
# real, once per MODE, through the same runpod.serverless.start patch gpu/azure_runner.py
# uses to import a worker safely -- not a hand-copied snippet of the scale/missing_keys
# assertions below, which would silently stop proving anything the day handler.py changes.
#
# A file, not an inline RUN: this used to be two separate checks here, each its own
# `python -c "..."`. The RIFE one broke ACR's build -- its continuation line (after the `\`
# that followed the sys.path.insert call) did the IFNet import from train_log.IFNet_HDv3 as
# its very first words, and ACR's dependency scanner misreads a RUN continuation line
# starting with that word as an attempted FROM instruction, failing the build before Docker
# itself ever runs it (the identical failure tts/build_check.py's own docstring records for
# a different lane -- deliberately not reproduced verbatim here, to keep this exact comment
# from being the thing that trips the same scanner). build_check.py proves the same two
# things -- "RIFE v4.26 loads OK, <n> tensors" with an empty missing_keys, and both
# Real-ESRGAN checkpoints report the scale the handler indexes them by -- by running the
# real code.
COPY build_check.py /build_check.py
RUN python /build_check.py

# the handler encodes with /usr/bin/ffmpeg (-c:v libx264 on both the upscale-video and
# interpolate paths): fail the build, not a billed GPU, if the system binary can't do h264.
# Plain grep, not `grep -q`: with pipefail on, -q closes its read end the moment it matches
# and SIGPIPEs ffmpeg mid-write (exit 141) before it finishes listing encoders -- a real
# failure on this exact check elsewhere in this repo (audio-clean, run ch2g, 2026-10-17).
# Plain grep drains the whole pipe, so ffmpeg always exits 0, and the matched line doubles
# as the build-log proof.
RUN /usr/bin/ffmpeg -hide_banner -encoders 2>/dev/null | grep libx264

CMD ["python", "-u", "/handler.py"]
