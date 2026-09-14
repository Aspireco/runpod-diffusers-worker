# Real-ESRGAN upscaling + RIFE frame interpolation. Both sets of weights are baked in:
# ~160MB total, small enough that baking beats a first-boot download and removes the
# runtime network dependency (and the Model Caching setup) entirely.
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
# the handler can switch on `scale` at import time -- FlashBoot snapshots the idle process,
# so anything loaded inside the handler is re-paid on every cold start.
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

# Worth more than the syntax check. The v4.26 checkpoint is DDP-saved, so every key carries
# a "module." prefix, and upstream loads it with strict=False -- get the prefix wrong and
# load_state_dict matches nothing, raises nothing, and the endpoint serves untrained mush.
# Asserting an empty missing_keys here turns that into a build failure.
RUN python -c "import sys, torch; sys.path.insert(0, '/opt/rife'); \
from train_log.IFNet_HDv3 import IFNet; \
sd = torch.load('/opt/rife/train_log/flownet.pkl', map_location='cpu', weights_only=True); \
sd = {k.replace('module.', '', 1): v for k, v in sd.items() if k.startswith('module.')}; \
r = IFNet().load_state_dict(sd, strict=False); \
assert not r.missing_keys, r.missing_keys[:5]; \
print('RIFE v4.26 loads OK,', len(sd), 'tensors')"

# Same idea for the upscalers: prove spandrel recognises both checkpoints and reports the
# scale the handler indexes them by, rather than discovering it on the first job.
RUN python -c "from spandrel import ImageModelDescriptor, ModelLoader; \
[print(f, '-> scale', ModelLoader().load_from_file('/opt/weights/' + f).scale) \
 for f in ('RealESRGAN_x2plus.pth', 'RealESRGAN_x4plus.pth')]; \
print('Real-ESRGAN weights load OK')"

CMD ["python", "-u", "/handler.py"]
