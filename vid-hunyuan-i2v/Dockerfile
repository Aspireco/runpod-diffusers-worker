# HunyuanVideo-I2V (diffusers port hunyuanvideo-community/HunyuanVideo-I2V, 43.6GB): a second
# top-tier image-to-video opinion alongside vid-wan-a14b. Azure GPU fleet lane `hunyuan-i2v`
# (A100 80GB PCIe). Weights are NOT in the image: they are staged on the Azure Files share by
# the fetch job (gpu/manifests/hunyuan-i2v.tsv) and mounted at /runpod-volume. This used to be a
# RunPod worker that pulled weights from RunPod's own Model Caching at runtime (MODEL_ID env var,
# HF cache glob-search) -- that mechanism does not exist on Azure, so this Dockerfile and
# handler.py now follow the same flat-directory convention every other lane in gpu/manifests/
# uses (see vid-wan-animate, vid-ltx25). Full licence table and VRAM reasoning in README.md.
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# The torch280 base ships HF_HUB_ENABLE_HF_TRANSFER=1 without the hf_transfer package, so the
# first runtime download from HuggingFace dies with "Fast download using 'hf_transfer' is
# enabled ... but 'hf_transfer' package is not available". Found live on the music lane
# 2026-09-14; every lane on this base inherits it. Moot here too (HF_HUB_OFFLINE=1 below means
# nothing downloads at runtime at all), but turning it off costs nothing.
ENV HF_HUB_ENABLE_HF_TRANSFER=0 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HUNYUAN_I2V_DIR="/runpod-volume/hunyuan-i2v" \
    OFFLOAD="model" \
    OUTPUT_FPS=24 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --ignore-installed cryptography
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Pin the base image's torch as constraints (see vid-wan-a14b/Dockerfile for the full story):
# left unpinned, diffusers/accelerate's bare "torch" dependency resolves to a CUDA 13 wheel from
# PyPI, which needs driver >= 580 and crash-loops with an empty log before Python even starts.
RUN python -c "import torch, torchvision, torchaudio; \
open('/tmp/constraints.txt','w').write(''.join(f'{n}=={v}\n' for n,v in \
[('torch',torch.__version__),('torchvision',torchvision.__version__),('torchaudio',torchaudio.__version__)]))" \
    && cat /tmp/constraints.txt

# diffusers >= 0.33 is the floor: HunyuanVideoImageToVideoPipeline does not exist before it, and
# the repo's model_index.json declares _diffusers_version 0.33.0.dev0. Unchanged from the
# original RunPod worker -- this lane is plain diffusers with no vendored model code, so there is
# no deep-internals coupling that would argue for pinning tighter (contrast workers/threed-hunyuan3d,
# which pins much closer to its upstream requirements.txt because it reaches into diffusers
# internals directly).
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

# Fail the BUILD, not a billed GPU: a file, not RUN python -c "..." continuation lines -- ACR's
# dependency scanner reads any line starting with `from` as a FROM instruction (vid-wan-a14b,
# 2026-10-17).
COPY build_check.py /build_check.py
RUN python /build_check.py imports
RUN python /build_check.py pipeline

COPY handler.py /handler.py
RUN python -c "import ast; ast.parse(open('/handler.py').read()); print('handler parses OK')"

CMD ["python", "-u", "/handler.py"]
