# Chatterbox Multilingual TTS. Weights (~3.5GB) are fetched on first boot rather than
# baked: they're small enough that the pull is quick, and Chatterbox contacts the hub even
# with a warm cache, so forcing offline mode here would crash-loop it.
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf \
    HF_HUB_DISABLE_TELEMETRY=1

# cryptography arrives via apt with no pip record, so pip cannot uninstall it for runpod's
# dependency chain. Shadow it instead of trying to remove it.
RUN pip install --no-cache-dir --ignore-installed cryptography

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Pin the base image's torch so chatterbox's own torch requirement cannot pull a CUDA 13
# wheel. Those need driver >= 580; Runpod hosts run 570/575 and CUDA init then dies before
# a single log line, which reads as a crash-loop with no cause.
RUN python -c "import torch, torchvision, torchaudio; \
open('/tmp/constraints.txt','w').write(''.join(f'{n}=={v}\n' for n,v in \
[('torch',torch.__version__),('torchvision',torchvision.__version__),('torchaudio',torchaudio.__version__)]))" \
    && cat /tmp/constraints.txt

# --only-binary refuses source builds entirely. Four earlier attempts died compiling from
# source: first an old pinned numpy hitting pkgutil.ImpImporter (removed in Python 3.12),
# then metadata-generation-failed once --no-build-isolation took away the isolated build
# env that had been supplying those build deps. Wheels-only sidesteps both.
#
# transformers is deliberately NOT pinned to the house >=4.50.3 here: chatterbox targets an
# older one and fails at runtime with "Could not import module 'LlamaModel'" because its
# lazy import shim can't resolve the class against the newer API. Let chatterbox choose;
# torch stays pinned by the constraints file, which is the dependency that breaks CUDA.
RUN pip install --no-cache-dir -c /tmp/constraints.txt --only-binary=:all: \
      "chatterbox-tts" "accelerate" \
    && pip install --no-cache-dir -c /tmp/constraints.txt \
      "huggingface-hub<1.0" \
      "runpod>=1.10.1"

# Fail the BUILD, not a billing GPU, if torch was swapped or the import that broke at
# runtime last time still breaks.
RUN python -c "import torch; v=torch.__version__; print('torch', v); assert 'cu13' not in v, f'torch replaced: {v}'"
RUN python -c "import transformers; print('transformers', transformers.__version__); \
from transformers import LlamaModel; print('LlamaModel import OK')"

COPY handler.py /handler.py
RUN python -c "import ast; ast.parse(open('/handler.py').read()); print('handler parses OK')"

CMD ["python", "-u", "/handler.py"]
