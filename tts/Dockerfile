# Chatterbox Multilingual TTS. Weights (~3.5GB) are fetched on first boot rather than
# baked: they're small enough that the pull is quick, and Chatterbox contacts the hub even
# with a warm cache, so forcing offline mode here would crash-loop it.
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf \
    HF_HUB_DISABLE_TELEMETRY=1

# Shadow the apt-installed cryptography pip cannot uninstall.
RUN pip install --no-cache-dir --ignore-installed cryptography

# One of chatterbox's transitive deps still ships a legacy setup.py that calls
# pkgutil.ImpImporter, removed in Python 3.12 -- the build dies with
# "module 'pkgutil' has no attribute 'ImpImporter'" unless setuptools is current.
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# torch is left exactly as the base image ships it — reinstalling pulls CUDA 13 wheels
# needing driver >= 580 while Runpod hosts run 570/575, and CUDA init dies with no logs.
# Install numpy and cython up front, then build chatterbox WITHOUT build isolation.
# Otherwise pip spins up an isolated build env, tries to compile an old pinned numpy from
# source, and dies on "pkgutil has no attribute ImpImporter" — Python 3.12 removed it and
# that numpy's legacy setup.py still calls it. --no-build-isolation reuses the numpy we
# just installed as a binary wheel instead of compiling anything.
# Pin the base image's torch so chatterbox's own torch requirement cannot pull a CUDA 13
# wheel. Those need driver >= 580; Runpod hosts run 570/575 and CUDA init then dies before
# a single log line, which reads as a crash-loop with no cause.
RUN python -c "import torch, torchvision, torchaudio; \
open('/tmp/constraints.txt','w').write(''.join(f'{n}=={v}\n' for n,v in \
[('torch',torch.__version__),('torchvision',torchvision.__version__),('torchaudio',torchaudio.__version__)]))" \
    && cat /tmp/constraints.txt

RUN pip install --no-cache-dir -c /tmp/constraints.txt "numpy>=1.26,<3" "cython"

# Deliberately NOT pinning transformers here. The house pin is >=4.50.3, but chatterbox
# targets an older transformers and fails on the newer one with
#   ModuleNotFoundError: Could not import module 'LlamaModel'
# — its lazy import shim can't resolve the class against the changed API. Let chatterbox
# resolve its own transformers; torch stays pinned by the constraints file, which is the
# dependency that actually breaks CUDA if it moves.
RUN pip install --no-cache-dir --no-build-isolation -c /tmp/constraints.txt \
      "chatterbox-tts" "accelerate" \
    && pip install --no-cache-dir -c /tmp/constraints.txt \
      "huggingface-hub<1.0" \
      "runpod>=1.10.1"

# Fail the build if the import that just broke at runtime still breaks.
RUN python -c "import transformers; print('transformers', transformers.__version__); \
from transformers import LlamaModel; print('LlamaModel import OK')"

RUN python -c "import torch; v=torch.__version__; print('torch', v); \
assert 'cu13' not in v, f'torch was replaced: {v}'"

COPY handler.py /handler.py
RUN python -c "import ast; ast.parse(open('/handler.py').read()); print('handler parses OK')"

CMD ["python", "-u", "/handler.py"]
