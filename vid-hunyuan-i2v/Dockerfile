# HunyuanVideo-I2V worker. 43.6GB of weights come from Runpod's Model Caching at runtime,
# so this image stays small, the lane needs no network volume, and -- unlike vid-wan-a14b --
# it is not pinned to a single datacentre.
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# The torch280 base ships HF_HUB_ENABLE_HF_TRANSFER=1 without the hf_transfer package,
# so the first runtime download from HuggingFace dies with "Fast download using
# 'hf_transfer' is enabled ... but 'hf_transfer' package is not available". Found live on
# the music lane 2026-09-14; every lane on this base inherits it. Turning the fast path
# off costs nothing -- it was never installed to begin with.
ENV HF_HUB_ENABLE_HF_TRANSFER=0

ENV MODEL_ID="hunyuanvideo-community/HunyuanVideo-I2V" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    OFFLOAD="model" \
    OUTPUT_FPS=24 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --ignore-installed cryptography
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Pin the base image's torch as constraints. diffusers and accelerate declare a bare torch
# dependency, and default PyPI now serves CUDA 13 wheels needing driver >= 580 while Runpod
# hosts run 570/575 -- CUDA init then dies before Python writes a line.
RUN python -c "import torch, torchvision, torchaudio; \
open('/tmp/constraints.txt','w').write(''.join(f'{n}=={v}\n' for n,v in \
[('torch',torch.__version__),('torchvision',torchvision.__version__),('torchaudio',torchaudio.__version__)]))" \
    && cat /tmp/constraints.txt

# diffusers >= 0.33 is the floor: HunyuanVideoImageToVideoPipeline does not exist before it,
# and the repo's model_index.json declares _diffusers_version 0.33.0.dev0.
RUN pip install --no-cache-dir -c /tmp/constraints.txt \
      "diffusers>=0.33" \
      "transformers>=4.50.3,<5" \
      "huggingface-hub<1.0" \
      "accelerate" \
      "sentencepiece" \
      "protobuf" \
      "ftfy" \
      "imageio" \
      "runpod>=1.10.1"

RUN python -c "import torch; v=torch.__version__; print('torch', v); assert 'cu13' not in v, f'torch replaced: {v}'"
RUN python -c "\
import diffusers; print('diffusers', diffusers.__version__); \
from diffusers import HunyuanVideoImageToVideoPipeline; \
from diffusers.models import HunyuanVideoTransformer3DModel; \
print('Hunyuan I2V pipeline + transformer classes present')"

COPY handler.py /handler.py
RUN python -c "import ast; ast.parse(open('/handler.py').read()); print('handler parses OK')"

CMD ["python", "-u", "/handler.py"]
