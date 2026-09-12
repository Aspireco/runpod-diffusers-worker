# Wan 2.2 I2V-A14B worker. Unlike every other lane in this repo, the weights do NOT come
# from Model Caching - at 126.2GB the repo is far past the 35-60GB band that is sized for.
# They are pre-staged on network volume yweuz29h2k (EUR-IS-1) by vid-wan-a14b/load_weights.sh,
# and the endpoint must attach that volume. The image itself stays small.
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# No HF_HUB_OFFLINE here, deliberately. The other workers set it because their weights are
# genuinely local; this one resolves a plain directory path and never calls the Hub, so the
# flag would only buy the misleading "outgoing traffic has been disabled" error if the
# volume were ever missing. A clear "weights not found" from resolve_model_dir beats that.
ENV MODEL_ID="Wan-AI/Wan2.2-I2V-A14B-Diffusers" \
    MODEL_DIR="/runpod-volume/wan22-i2v-a14b" \
    HF_HUB_DISABLE_TELEMETRY=1 \
    OFFLOAD="model" \
    DTYPE="bf16" \
    OUTPUT_FPS=16 \
    PYTHONUNBUFFERED=1

# ffmpeg encodes the frame stream; frames pipe straight in rather than landing on disk.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --ignore-installed cryptography
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Pin the base image's torch as constraints. diffusers, accelerate and torchao all declare
# a bare torch dependency, and pip satisfies it from default PyPI - which since 2.11 means
# CUDA 13 wheels needing driver >= 580. Runpod hosts run 570/575, so CUDA init dies before
# Python writes a line and the worker crash-loops with empty logs.
RUN python -c "import torch, torchvision, torchaudio; \
open('/tmp/constraints.txt','w').write(''.join(f'{n}=={v}\n' for n,v in \
[('torch',torch.__version__),('torchvision',torchvision.__version__),('torchaudio',torchaudio.__version__)]))" \
    && cat /tmp/constraints.txt

# diffusers >= 0.35 is a hard floor, not a preference: Wan 2.2's MoE needs the
# transformer_2 slot in WanImageToVideoPipeline and AutoencoderKLWan, and the repo's own
# model_index.json declares _diffusers_version 0.35.0.dev0. Older releases load the first
# expert, ignore the second, and produce degraded video with no error.
# torchao supplies the fp8 arm of the benchmark; it is optional at runtime and the handler
# falls back to bf16 if the import fails, so a torchao break cannot take the lane down.
RUN pip install --no-cache-dir -c /tmp/constraints.txt \
      "diffusers>=0.35" \
      "transformers>=4.50.3,<5" \
      "huggingface-hub<1.0" \
      "accelerate" \
      "sentencepiece" \
      "protobuf" \
      "ftfy" \
      "imageio" \
      "torchao" \
      "runpod>=1.10.1"

# Fail the BUILD, not a billing GPU, if torch was swapped or the MoE classes are absent.
RUN python -c "import torch; v=torch.__version__; print('torch', v); assert 'cu13' not in v, f'torch replaced: {v}'"
RUN python -c "\
import diffusers, inspect; print('diffusers', diffusers.__version__); \
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline; \
p = set(inspect.signature(WanImageToVideoPipeline.__init__).parameters); \
assert 'transformer_2' in p, f'diffusers {diffusers.__version__} has no MoE second expert: {sorted(p)}'; \
print('Wan MoE pipeline OK, dual-expert slot present')"

COPY handler.py /handler.py
RUN python -c "import ast; ast.parse(open('/handler.py').read()); print('handler parses OK')"

CMD ["python", "-u", "/handler.py"]
