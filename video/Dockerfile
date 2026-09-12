# Cosmos 3 video worker. No weights baked in — Runpod's Model Caching mounts the 35GB repo
# at runtime, which keeps this image small and avoids a network volume entirely.
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV MODEL_ID="nvidia/Cosmos3-Nano" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    CPU_OFFLOAD=1 \
    OUTPUT_FPS=24 \
    PYTHONUNBUFFERED=1

# ffmpeg encodes the frame stream; frames pipe straight in rather than landing on disk.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --ignore-installed cryptography
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Pin the base image's torch as constraints. diffusers, accelerate and friends all declare a
# bare torch dependency, and pip satisfies it from default PyPI — which since 2.11 means
# CUDA 13 wheels needing driver >= 580. Runpod hosts run 570/575, so CUDA init dies before
# Python writes a line and the worker crash-loops with empty logs.
RUN python -c "import torch, torchvision, torchaudio; \
open('/tmp/constraints.txt','w').write(''.join(f'{n}=={v}\n' for n,v in \
[('torch',torch.__version__),('torchvision',torchvision.__version__),('torchaudio',torchaudio.__version__)]))" \
    && cat /tmp/constraints.txt

# Cosmos3 declares _diffusers_version 0.37.1 and needs the Cosmos3OmniDiffusersPipeline
# class, which older diffusers does not ship — hence the floor rather than the house >=0.31.
RUN pip install --no-cache-dir -c /tmp/constraints.txt \
      "diffusers>=0.37.1" \
      "transformers>=4.50.3,<5" \
      "huggingface-hub<1.0" \
      "accelerate" \
      "sentencepiece" \
      "protobuf" \
      "imageio" \
      "runpod>=1.10.1"

# Fail the BUILD, not a billing GPU, if torch was swapped or the pipeline class is absent.
RUN python -c "import torch; v=torch.__version__; print('torch', v); assert 'cu13' not in v, f'torch replaced: {v}'"
RUN python -c "import diffusers; print('diffusers', diffusers.__version__); \
assert hasattr(diffusers, 'DiffusionPipeline')"

COPY handler.py /handler.py
RUN python -c "import ast; ast.parse(open('/handler.py').read()); print('handler parses OK')"

CMD ["python", "-u", "/handler.py"]
