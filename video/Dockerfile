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

# Cosmos3-Nano's model_index.json says _diffusers_version 0.37.1, but that is the version
# that SAVED the repo, not one that can load it. The class it actually needs,
# Cosmos3OmniPipeline, first shipped in diffusers v0.39.0 -- so the floor is 0.39, not 0.37.
# (The repo names its class "Cosmos3OmniDiffusersPipeline", which no diffusers release has
# ever exported; handler.py maps that wrong name onto the real one.)
# cosmos_guardrail is deliberately NOT installed; the handler passes
# enable_safety_checker=False instead. It cannot coexist with this image:
# cosmos-guardrail 0.3.x requires transformers>=5.0.0 against the house pin of <5,
# and 0.1.0 requires torch==2.6.0 against this base's 2.8 -- no satisfiable version
# exists, which pip reports as ResolutionImpossible.
#
# Cosmos3OmniPipeline.__init__ takes `enable_safety_checker: bool = True` and only
# constructs CosmosSafetyChecker when it is on, so False skips the import entirely.
# safety_checker is already listed in the pipeline's _optional_components.
#
# This DOES disable NVIDIA's content filter. Acceptable for a lane that renders
# first-party marketing footage from prompts we write ourselves, but it is a
# deliberate trade recorded in DECISIONS.md rather than an oversight.
RUN pip install --no-cache-dir -c /tmp/constraints.txt \
      "diffusers>=0.39.0" \
      "transformers>=4.50.3,<5" \
      "huggingface-hub<1.0" \
      "accelerate" \
      "sentencepiece" \
      "protobuf" \
      "imageio" \
      "runpod>=1.10.1"

# Fail the BUILD, not a billing GPU, if torch was swapped or the pipeline class is absent.
RUN python -c "import torch; v=torch.__version__; print('torch', v); assert 'cu13' not in v, f'torch replaced: {v}'"
RUN python -c "import diffusers; print('diffusers', diffusers.__version__); assert hasattr(diffusers, 'Cosmos3OmniPipeline'), 'diffusers %s lacks Cosmos3OmniPipeline (needs >=0.39.0)' % diffusers.__version__"

COPY handler.py /handler.py
RUN python -c "import ast; ast.parse(open('/handler.py').read()); print('handler parses OK')"

CMD ["python", "-u", "/handler.py"]
