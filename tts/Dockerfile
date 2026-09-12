# Chatterbox Multilingual TTS. Weights (~3.5GB) are fetched on first boot rather than
# baked: they're small enough that the pull is quick, and Chatterbox contacts the hub even
# with a warm cache, so forcing offline mode here would crash-loop it.
# BASE PINNED TO torch 2.6.0 ON PURPOSE -- do not "upgrade" this line.
# chatterbox-tts hard-pins torch==2.6.0 at EVERY published version (0.1.0-0.1.3 and 0.1.7
# pin torch==2.6.0; 0.1.4-0.1.6 instead pin numpy<1.26.0). The torch280 base ships
# 2.8.0+cu128 and numpy 2.1.2, so pip has no satisfiable version at all and reports
# ResolutionImpossible. Six build attempts were spent tuning pip flags against what is a
# hard == pin -- no flag can fix that; only a base whose torch matches can.
# cu1281 is kept deliberately: Runpod hosts run driver 570/575, and a CUDA 13 build needs
# >= 580, which dies at CUDA init before Python emits a line.
FROM runpod/pytorch:1.2.0-cu1281-torch260-ubuntu2404

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf \
    HF_HUB_DISABLE_TELEMETRY=1

# cryptography arrives via apt with no pip record, so pip cannot uninstall it for runpod's
# dependency chain. Shadow it instead of trying to remove it.
RUN pip install --no-cache-dir --ignore-installed cryptography

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Pin the base image's torch so nothing in the dependency tree can pull a CUDA 13 wheel.
# Those need driver >= 580; Runpod hosts run 570/575 and CUDA init then dies before a single
# log line, which reads as a crash-loop with no cause. With the torch260 base this constraint
# now AGREES with chatterbox's torch==2.6.0 pin instead of contradicting it -- PEP 440 treats
# the local label in 2.6.0+cu128 as satisfying a bare ==2.6.0.
RUN python -c "import torch, torchvision, torchaudio; \
open('/tmp/constraints.txt','w').write(''.join(f'{n}=={v}\n' for n,v in \
[('torch',torch.__version__),('torchvision',torchvision.__version__),('torchaudio',torchaudio.__version__)]))" \
    && cat /tmp/constraints.txt

# --only-binary is scoped to numpy, NOT :all:. Four earlier attempts died compiling numpy
# from source (an old pin hitting pkgutil.ImpImporter, removed in Python 3.12; then
# metadata-generation-failed once --no-build-isolation removed the build env supplying its
# build deps). Attempt five used --only-binary=:all: and died differently: that flag also
# applies to torch, and the constraints file above pins torch to the base image's LOCAL
# version (e.g. 2.8.0+cu128), published on download.pytorch.org and never on PyPI.
# Wheels-only then demands a PyPI distribution that cannot exist, and pip reports
# "ResolutionImpossible ... no matching distributions available for your environment:
# numpy, torch". Scoping the flag to numpy keeps the source-build fix while letting torch
# be satisfied by the copy already in the base image -- which is exactly what video/ and
# relight/ do, and both of those build green.
#
# transformers is deliberately NOT pinned to the house >=4.50.3 here: chatterbox targets an
# older one and fails at runtime with "Could not import module 'LlamaModel'" because its
# lazy import shim can't resolve the class against the newer API. Let chatterbox choose;
# torch stays pinned by the constraints file, which is the dependency that breaks CUDA.
# huggingface-hub is deliberately NOT pinned. The rebase to torch 2.6.0 worked and
# chatterbox 0.1.7 installed cleanly WITH transformers 5.2.0 and huggingface-hub
# 1.31.0 -- then a stale `huggingface-hub<1.0` pin here DOWNGRADED it to 0.36.2 and
# broke transformers 5.x, which imports is_offline_mode from the newer hub API:
#   ImportError: cannot import name 'is_offline_mode' from 'huggingface_hub'
# The pin was protecting against an older breakage and is now the cause of one.
RUN pip install --no-cache-dir -c /tmp/constraints.txt --only-binary=numpy \
      "chatterbox-tts" "accelerate" \
    && pip install --no-cache-dir -c /tmp/constraints.txt \
      "runpod>=1.10.1"

# Fail the BUILD, not a billing GPU, if torch was swapped or the import that broke at
# runtime last time still breaks.
RUN python -c "import torch; v=torch.__version__; print('torch', v); assert 'cu13' not in v, f'torch replaced: {v}'"
RUN python -c "import transformers; print('transformers', transformers.__version__); \
from transformers import LlamaModel; print('LlamaModel import OK')"

COPY handler.py /handler.py
RUN python -c "import ast; ast.parse(open('/handler.py').read()); print('handler parses OK')"

CMD ["python", "-u", "/handler.py"]
