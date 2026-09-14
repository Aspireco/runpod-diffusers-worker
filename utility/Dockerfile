# BiRefNet cutouts. ~450MB of weights, fetched on first boot.
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# The torch280 base ships HF_HUB_ENABLE_HF_TRANSFER=1 without the hf_transfer package,
# so the first runtime download from HuggingFace dies with "Fast download using
# 'hf_transfer' is enabled ... but 'hf_transfer' package is not available". Found live on
# the music lane 2026-09-14; every lane on this base inherits it. Turning the fast path
# off costs nothing -- it was never installed to begin with.
ENV HF_HUB_ENABLE_HF_TRANSFER=0

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf \
    MODEL_ID=ZhengPeng7/BiRefNet \
    INPUT_RES=1024 \
    HF_HUB_DISABLE_TELEMETRY=1

RUN pip install --no-cache-dir --ignore-installed cryptography
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# BiRefNet loads through trust_remote_code, which needs timm and einops.
#
# The constraints file is load-bearing, not belt-and-braces. timm and kornia both declare a
# bare `torch` dependency, so pip is free to "satisfy" it by pulling a newer torch from
# default PyPI -- which since 2.11 means CUDA 13 wheels requiring driver >= 580. Runpod
# hosts run 570/575, so CUDA init dies before Python writes a single line and the worker
# crash-loops with completely empty container logs. Pinning the already-installed versions
# as constraints makes that upgrade impossible.
RUN python -c "import torch, torchvision, torchaudio; \
open('/tmp/constraints.txt','w').write(''.join(f'{n}=={v}\n' for n,v in \
[('torch',torch.__version__),('torchvision',torchvision.__version__),('torchaudio',torchaudio.__version__)]))" \
    && cat /tmp/constraints.txt

RUN pip install --no-cache-dir -c /tmp/constraints.txt \
      "transformers>=4.50.3,<5" \
      "huggingface-hub<1.0" \
      "timm" \
      "einops" \
      "kornia" \
      "runpod>=1.10.1"

# Prove torch still points at the CUDA 12.8 build the base image shipped. If a dependency
# dragged in a cu13 wheel anyway, fail the BUILD here rather than crash-looping on a GPU.
RUN python -c "import torch; v=torch.__version__; print('torch', v); \
assert 'cu13' not in v and not v.startswith('2.1'), f'torch was replaced: {v}'"

COPY handler.py /handler.py
RUN python -c "import ast; ast.parse(open('/handler.py').read()); print('handler parses OK')"

CMD ["python", "-u", "/handler.py"]
