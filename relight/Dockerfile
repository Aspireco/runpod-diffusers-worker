# IC-Light relighting / scene compositing. ~7.5GB of weights, fetched on first boot:
# the SD1.5 base (~4GB) plus two 1.72GB IC-Light offset files. Give the endpoint at least
# 30GB of container disk, or point Runpod Model Caching at the two repos.
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
    BASE_MODEL=stablediffusionapi/realistic-vision-v51 \
    IC_LIGHT_REPO=lllyasviel/ic-light \
    ENABLE_FC=1 \
    BASE_SIDE=640

# cryptography ships via apt in the base image with no pip record, so pip refuses to
# upgrade it for runpod's dependency chain ("Cannot uninstall cryptography ...
# uninstall-no-record-file"). Shadow it rather than trying to remove it. Must be first.
RUN pip install --no-cache-dir --ignore-installed cryptography
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Pins, each one load-bearing:
#   diffusers <0.36        -- this worker hooks unet.forward and smuggles conditioning
#                             through cross_attention_kwargs, and hands raw 4-channel
#                             latents to the img2img pipeline. Both are internals, not
#                             API, so the upper bound is real. Verified against 0.35.1.
#   transformers <5        -- the 5.x major breaks diffusers imports
#   huggingface-hub <1.0   -- likewise the 1.x major
#   runpod >=1.10.1        -- 1.7.11 through 1.10.0 corrupt job tracking
#
# torch and torchvision are left exactly as the base image ships them. Reinstalling from
# default PyPI pulls CUDA 13 wheels needing driver >=580 while Runpod hosts run 570/575,
# and CUDA init then dies before a single log line is written. accelerate's requirement is
# torch>=2.0.0, which the image's 2.8.0 already satisfies, so it will not drag one in.
#
# Nothing here is opencv/mmcv/lightning: libcom was evaluated for shadow generation and
# harmonisation and rejected because its requirements.txt hard-pins torch==2.6.0,
# torchvision==0.21.0 and transformers==4.44.2 and pulls the mmdet+mmpose+mmcv compile
# chain. handler.py does the shadow and detail passes in numpy/PIL instead.
RUN pip install --no-cache-dir \
      "diffusers>=0.31,<0.36" \
      "transformers>=4.50.3,<5" \
      "huggingface-hub<1.0" \
      "accelerate" \
      "safetensors" \
      "runpod>=1.10.1"

COPY handler.py /handler.py

# Fail the BUILD rather than the worker: a syntax error surfaces here, where the error is
# visible, instead of as a silent crash-loop with no container logs.
RUN python -c "import ast; ast.parse(open('/handler.py').read()); print('handler parses OK')"

CMD ["python", "-u", "/handler.py"]
