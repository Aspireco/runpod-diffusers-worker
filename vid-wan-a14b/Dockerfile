# Wan 2.2 I2V-A14B worker. Unlike every other lane in this repo, the weights do NOT come
# from Model Caching - at 126.2GB the repo is far past the 35-60GB band that is sized for.
# They are pre-staged on network volume yweuz29h2k (EUR-IS-1) by vid-wan-a14b/load_weights.sh,
# and the endpoint must attach that volume. The image itself stays small.
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# The torch280 base ships HF_HUB_ENABLE_HF_TRANSFER=1 without the hf_transfer package,
# so the first runtime download from HuggingFace dies with "Fast download using
# 'hf_transfer' is enabled ... but 'hf_transfer' package is not available". Found live on
# the music lane 2026-09-14; every lane on this base inherits it. Turning the fast path
# off costs nothing -- it was never installed to begin with.
ENV HF_HUB_ENABLE_HF_TRANSFER=0

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
RUN pip install --no-cache-dir -c /tmp/constraints.txt \
      "diffusers>=0.35" \
      "transformers>=4.50.3,<5" \
      "huggingface-hub<1.0" \
      "accelerate" \
      "sentencepiece" \
      "protobuf" \
      "ftfy" \
      "imageio" \
      "runpod>=1.10.1"

# Fail the BUILD, not a billing GPU, if torch was swapped or the MoE classes are absent.
RUN python -c "import torch; v=torch.__version__; print('torch', v); assert 'cu13' not in v, f'torch replaced: {v}'"
RUN python -c "\
import diffusers, inspect; print('diffusers', diffusers.__version__); \
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline; \
p = set(inspect.signature(WanImageToVideoPipeline.__init__).parameters); \
assert 'transformer_2' in p, f'diffusers {diffusers.__version__} has no MoE second expert: {sorted(p)}'; \
print('Wan MoE pipeline OK, dual-expert slot present')"


# torchao LAST, in its own layer, and pinned -- all three deliberate.
#
# An unpinned `torchao` broke this build once, and not in its own code: torchao declares NO
# torch dependency in its metadata, so pip installed the newest release (built for torch
# 2.9+) alongside the base image's torch 2.8. diffusers autodetects torchao and imports its
# integration, which reaches for `torch.nn.functional.ScalingType` -- absent in 2.8. The
# result was `cannot import name 'ScalingType'` raised from
# `diffusers.models.autoencoders.autoencoder_kl_wan`: an optional benchmark dependency
# silently broke EVERY diffusers import. The cu13 torch guard above passed throughout,
# because torch itself was never replaced -- a constraints file pins what pip resolves, it
# cannot pin what a dependency expects at runtime.
#
# 0.13.0 is the release built in the torch 2.8 window: torch 2.8.0 2025-08-06, torchao
# 0.13.0 2025-09-04, then torchao 0.14.0 2025-10-13 ships with torch 2.9.0 2025-10-15.
#
# It is installed after the diffusers check on purpose, so the core lane is proven sound
# before an optional dependency is allowed near it and any future break is unambiguously
# torchao's. The handler also try/excepts the import and reports the dtype that actually
# ran, so a runtime failure degrades to bf16 honestly instead of mislabelling the result.
RUN pip install --no-cache-dir -c /tmp/constraints.txt "torchao==0.13.0"

# The check that would have caught the original failure: does diffusers STILL import once
# torchao is present, and does the fp8 entry point actually resolve?
RUN python -c "\
import torch; assert 'cu13' not in torch.__version__, f'torch replaced: {torch.__version__}'; \
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline; \
import torchao; from torchao.quantization import float8_weight_only, quantize_; \
print('torchao', torchao.__version__, 'coexists with diffusers; fp8 arm available')"
COPY handler.py /handler.py
RUN python -c "import ast; ast.parse(open('/handler.py').read()); print('handler parses OK')"

CMD ["python", "-u", "/handler.py"]
