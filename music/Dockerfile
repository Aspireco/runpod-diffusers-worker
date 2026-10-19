# ACE-Step 1.5 (MIT, github.com/ace-step/ACE-Step-1.5) music generation, DiT-only turbo config.
# Weights come from the Azure Files share (see README.md "Weights" -- several GB, well past this
# repo's ~500MB bake budget); this image carries no model files at all.
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# The torch280 base ships HF_HUB_ENABLE_HF_TRANSFER=1 without the hf_transfer package, so a runtime
# HF download dies with "Fast download using 'hf_transfer' is enabled ... but 'hf_transfer' package
# is not available" (found live on this same lane, 2026-09-14). Turning it off costs nothing here --
# ACESTEP_CHECKPOINTS_DIR below makes every checkpoint a local-disk hit, never a download.
ENV HF_HUB_ENABLE_HF_TRANSFER=0 \
    PYTHONUNBUFFERED=1 \
    ACESTEP_CHECKPOINTS_DIR=/runpod-volume/acestep-v15/checkpoints \
    ACESTEP_DISABLE_TQDM=1

# ffmpeg: audio_utils.AudioSaver shells out to it for format conversion (confirmed by its own
# `import subprocess`); matches this repo's standing policy for any audio-touching lane.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --ignore-installed cryptography
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Pin the base image's own torch as constraints -- the same trick every worker in this repo uses.
# ACE-Step 1.5's own pyproject.toml declares torch==2.10.0+cu128 on Linux x86_64; pip would happily
# satisfy that with a CUDA 13 wheel needing driver >= 580, and Runpod/this base's hosts run 570/575,
# so CUDA init dies before a single log line (the exact failure mode the old v1 Dockerfile's own
# comment on this base already documented).
RUN python -c "import torch, torchvision, torchaudio; \
open('/tmp/constraints.txt','w').write(''.join(f'{n}=={v}\n' for n,v in \
[('torch',torch.__version__),('torchvision',torchvision.__version__),('torchaudio',torchaudio.__version__)]))" \
    && cat /tmp/constraints.txt

# --no-deps is deliberate, not a shortcut: ACE-Step 1.5's pyproject.toml lists `nano-vllm` as an
# unconditional Linux dependency, resolved only through uv's own `[tool.uv.sources]` (a git/local
# source) -- it does not exist on PyPI at all (checked directly, 2026-10-17: `pip download --no-deps
# nano-vllm` -> "No matching distribution found for nano-vllm"). A plain `pip install` of this
# project therefore fails outright, before anything GPU-related runs. The full dependency list also
# carries gradio, fastapi, lightning, tensorboard, peft, lycoris-lora, mlx/mlx-lm (Apple-only) and
# modelscope -- all for the Gradio UI, REST API server, LoRA/full training, and Apple Silicon, none
# of which this handler touches. Every one of those backends is imported lazily, inside the specific
# method that needs it, not at module import time -- confirmed by reading
# init_service_loader.py / init_service_loader_components.py (torchao quantization, diffusers'
# AutoencoderOobleck, transformers AutoModel/AutoTokenizer are all `from X import Y` lines inside
# functions, never at the top of a file) -- so installing only the package itself plus the handful
# of pure-inference packages below is enough for AceStepHandler + generate_music with no LM.
# Pinned to the commit the 3b8db3d-acestep15 image resolved (ACR run ch2u, 2026-10-18).
RUN pip install --no-cache-dir --no-deps "git+https://github.com/ace-step/ACE-Step-1.5.git@ca1e85fe9430179831e6bc6be790c332190a3866"

# The minimal set `acestep.handler` / `acestep.inference` need for DiT-only text2music (no LM, no
# quantization, no training, no Gradio/API server): transformers (AutoModel/AutoTokenizer, the DiT's
# own trust_remote_code=True model class, and the Qwen3-Embedding-0.6B text encoder), diffusers
# (AutoencoderOobleck VAE), accelerate (device placement), soundfile + torchaudio (audio I/O,
# torchaudio already pinned above), loguru (this project's own logger), einops (tensor ops used
# throughout its model code), scipy/numba/pywavelets/vector-quantize-pytorch (direct, non-extra
# pyproject deps touched by its signal-processing and repaint/lego paths), diskcache + toml (its own
# small utility deps). huggingface-hub<1.0 and transformers' own upper bound match every other worker
# in this repo; build_check.py below proves the actual import, not just this list.
RUN pip install --no-cache-dir -c /tmp/constraints.txt \
      "transformers>=4.51.0,<4.58.0" \
      "huggingface-hub<1.0" \
      "diffusers>=0.37.0" \
      "accelerate>=1.12.0" \
      "soundfile>=0.13.1" \
      "loguru>=0.7.3" \
      "einops>=0.8.1" \
      "scipy>=1.10.1" \
      "numba>=0.63.1" \
      "pywavelets>=1.9.0" \
      "vector-quantize-pytorch>=1.27.15" \
      "diskcache" \
      "toml" \
    && pip install --no-cache-dir -c /tmp/constraints.txt "runpod>=1.10.1"

RUN python -c "import torch; v=torch.__version__; print('torch', v); \
assert 'cu13' not in v, f'torch was replaced: {v}'"

COPY build_check.py /build_check.py
RUN python /build_check.py

COPY handler.py /handler.py
RUN python -c "import ast; ast.parse(open('/handler.py').read()); print('handler parses OK')"

CMD ["python", "-u", "/handler.py"]
