# TIER 2 CALIBRATION WORKER -- Coqui XTTS v2. NON-COMMERCIAL WEIGHTS.
#
# Nothing this image produces may be published. It exists to answer one question: how much
# quality does staying commercially clean actually cost on voice cloning? XTTS v2 was the
# default open cloner for about two years, so it is the right yardstick, and Chatterbox (MIT,
# endpoint kqow92p29gl9bn) is what it is measured against.
#
# LICENCE, PRECISELY:
#   weights  coqui/XTTS-v2        Coqui Public Model License (CPML), https://coqui.ai/cpml
#                                 NON-COMMERCIAL. And Coqui wound down as a company, so there
#                                 is no longer anyone to buy a commercial licence FROM. That
#                                 is what makes this permanent rather than merely pending.
#   code     coqui-tts (idiap)    MPL-2.0. Permissive. IRRELEVANT to the weights above.
#
# The code/weights split is the trap. `pip show coqui-tts` says MPL-2.0 and a reasonable
# person would conclude the model is fine to use. It is not. Licence lives with the
# checkpoint, not with the loader.
#
# The image name carries the tier (runpod-tier2-voice-worker) and the endpoint carries the
# t2- prefix, so both the registry and the publishing guard can refuse it on name alone,
# before anyone inspects a single asset.
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# HF_HUB_ENABLE_HF_TRANSFER=0 is load-bearing, not tidying. The base image ships it set to
# 1, and pinning huggingface-hub<1.0 below resolves an environment where the hf_transfer
# package is absent. The hub then RAISES instead of falling back to plain HTTP, which killed
# the weights download in half a second. Turned off rather than installing hf_transfer: one
# dependency fewer beats a marginally faster one-time 1.8GB pull.
#
# COQUI_TOS_AGREED=0 is also deliberate. Setting it to 1 is how the TTS library auto-accepts
# CPML on your behalf, and this worker must never accept a licence implicitly -- the weights
# are fetched directly below, from an ungated repo, with the licence text shipped alongside.
ENV TIER2_ENDPOINT_NAME="t2-xtts-v2" \
    XTTS_DIR="/models/xtts-v2" \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    PYTHONUNBUFFERED=1 \
    COQUI_TOS_AGREED=0

# ffmpeg/libsndfile for the audio path; git because coqui-tts pulls a couple of VCS-less
# deps that still want it present at build.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --ignore-installed cryptography
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Pin the base image's torch as constraints. coqui-tts declares a bare torch dependency and
# pip would satisfy it from default PyPI, which since 2.11 means CUDA 13 wheels needing
# driver >= 580. Runpod hosts run 570/575, so CUDA init dies before Python writes a line and
# the worker crash-loops with completely empty logs. Every install below carries -c.
# numpy is pinned alongside torch, not separately. coqui-tts pulls a deep scientific stack
# and any of it can drag numpy across the 1.x/2.x boundary, which breaks the base image's
# torch ABI just as surely as replacing torch itself would.
RUN python -c "import torch, torchvision, torchaudio, numpy; \
open('/tmp/constraints.txt','w').write(''.join(f'{n}=={v}\n' for n,v in \
[('torch',torch.__version__),('torchvision',torchvision.__version__),('torchaudio',torchaudio.__version__),('numpy',numpy.__version__)]))" \
    && cat /tmp/constraints.txt

# coqui-tts, NOT the original `TTS` package: TTS 0.22.0 caps at Python <3.12 and this base
# is Ubuntu 24.04 / Python 3.12, so the original resolves to nothing installable. The idiap
# fork is the maintained continuation. It changes the CODE licence, never the weights'.
#
# --prefer-binary, NOT --only-binary=:all:. The house rule prefers binary-only because
# SOURCE BUILDS of C extensions killed a worker five times over. But coqui-tts depends on
# num2words -> docopt, and docopt has NO wheel on PyPI at any version -- "(from versions:
# none)". --only-binary=:all: therefore cannot resolve this tree at all; it failed the first
# build in 5.7 seconds. docopt is a single pure-Python file with nothing to compile, so
# --prefer-binary keeps the protection that matters (wheels win wherever one exists) while
# letting the handful of pure-Python sdists through. The torch assert below is what actually
# enforces the rule the house rule exists to protect.
RUN pip install --no-cache-dir -c /tmp/constraints.txt --prefer-binary \
      "coqui-tts==0.27.5" \
      "huggingface-hub<1.0" \
      "runpod>=1.10.1"

# Fail the BUILD, not a billing GPU, if torch or numpy was swapped out from under us.
# This is the assert the --prefer-binary relaxation above is leaning on, so it checks the
# exact failure it is protecting against: a CUDA 13 wheel that needs driver >= 580 on hosts
# running 570/575, which dies before Python prints a line.
RUN python -c "import torch, numpy; v=torch.__version__; print('torch', v, '| numpy', numpy.__version__); \
assert 'cu13' not in v, f'torch replaced: {v}'; \
import torch.version as tv; print('cuda', tv.cuda); \
assert tv.cuda is None or not str(tv.cuda).startswith('13'), f'CUDA 13 build: {tv.cuda}'"
RUN python -c "from TTS.tts.models.xtts import Xtts; print('xtts importable')"

# Weights baked in rather than mounted through Runpod Model Caching. XTTS v2 is ~1.8GB, the
# TTS library wants a specific on-disk layout rather than an HF snapshot tree, and baking
# means the licence text ships inside the image next to the checkpoint it governs.
# HF_TOKEN is not needed: coqui/XTTS-v2 is ungated. It is non-commercial, not restricted --
# a distinction worth keeping straight, since ungated is what let it become the default.
RUN python -c "\
from huggingface_hub import snapshot_download; \
p = snapshot_download('coqui/XTTS-v2', local_dir='/models/xtts-v2', \
    allow_patterns=['config.json','model.pth','vocab.json','speakers_xtts.pth','LICENSE.txt','samples/en_sample.wav']); \
print('weights at', p)"

# The licence text must be present in the image. If Coqui's repo ever loses it, this build
# fails rather than quietly shipping weights whose terms nobody can read.
RUN test -s /models/xtts-v2/LICENSE.txt \
    && head -c 400 /models/xtts-v2/LICENSE.txt \
    && echo "" && echo "CPML present"

COPY handler.py /handler.py
RUN python -c "import ast; ast.parse(open('/handler.py').read()); print('handler parses OK')"

# THE CHECK THAT MATTERS. Run the handler's tier-tagging path at build time, on CPU, with
# the model import stubbed out. A broken tagger fails the BUILD here -- where the log is
# readable -- instead of shipping a worker that emits untagged non-commercial audio. The
# handler also re-runs this at import, so a worker whose tagging is broken never boots.
RUN python -c "\
import ast, sys, types, json;\
src = open('/handler.py').read();\
tree = ast.parse(src);\
cut = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and getattr(n.value.func, 'id', '') == '_selfcheck');\
mod = types.ModuleType('tagcheck');\
exec(compile(ast.Module(body=tree.body[:cut+1], type_ignores=[]), 'handler.py', 'exec'), mod.__dict__);\
import numpy as np;\
blob = mod.wav_with_tag(np.zeros(256, dtype='float32'), 24000, mod.build_tag());\
info = mod._read_riff_info(blob);\
tag = json.loads(info['ICMT']);\
assert tag['tier'] == '2', tag;\
assert tag['commercial_use'] == 'FORBIDDEN', tag;\
assert 'marketing_os_tier' in info['ISFT'], info;\
assert b'RIFF' == blob[:4] and b'data' in blob;\
print('BUILD-TIME TIER TAG CHECK PASSED:', tag['tier'], tag['license'])"

CMD ["python", "-u", "/handler.py"]
