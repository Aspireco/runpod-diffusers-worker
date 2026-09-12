# ACE-Step music generation. Weights download on first boot into ACE_CHECKPOINT_DIR.
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf \
    ACE_CHECKPOINT_DIR=/opt/ace-checkpoints \
    HF_HUB_DISABLE_TELEMETRY=1

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --ignore-installed cryptography
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# ACE-Step is not published to PyPI — "pip install acestep" finds no distribution at all.
# It installs from source. Pinned to a commit-less main because the project tags rarely.
# Pin the torch the base image already has. ACE-Step installs from git and declares its own
# torch requirement, which pip would happily satisfy with a CUDA 13 wheel needing driver
# >= 580 -- Runpod hosts run 570/575, and the worker then crash-loops with empty logs.
RUN python -c "import torch, torchvision, torchaudio; \
open('/tmp/constraints.txt','w').write(''.join(f'{n}=={v}\n' for n,v in \
[('torch',torch.__version__),('torchvision',torchvision.__version__),('torchaudio',torchaudio.__version__)]))" \
    && cat /tmp/constraints.txt

RUN pip install --no-cache-dir -c /tmp/constraints.txt "numpy>=1.26,<3"
RUN pip install --no-cache-dir --no-build-isolation -c /tmp/constraints.txt \
      "git+https://github.com/ace-step/ACE-Step.git" \
    && pip install --no-cache-dir -c /tmp/constraints.txt \
      "transformers>=4.50.3,<5" \
      "huggingface-hub<1.0" \
      "runpod>=1.10.1"

RUN python -c "import torch; v=torch.__version__; print('torch', v); \
assert 'cu13' not in v, f'torch was replaced: {v}'"

RUN mkdir -p /opt/ace-checkpoints

COPY handler.py /handler.py
RUN python -c "import ast; ast.parse(open('/handler.py').read()); print('handler parses OK')"

CMD ["python", "-u", "/handler.py"]
