"""Run at image build time: fail the BUILD, not a billed GPU (a file, because ACR's scanner
misreads a Dockerfile RUN continuation line that begins with `from` as a FROM instruction --
see tts/build_check.py's own docstring for the real failure that taught this repo that).

Constructs (not loads) all three MODEL_ID variants this image supports, via AutoConfig +
AutoModelForImageSegmentation.from_config: that fetches each repo's tiny config.json plus its
remote-code birefnet.py/BiRefNet_config.py (tens of KB total, confirmed live 2026-10-17 --
ZhengPeng7/BiRefNet's own birefnet.py is 91,896 bytes), and builds the architecture with
random weights, never the real 444MB-885MB model.safetensors. That proves the one real
build-time risk -- BiRefNet's trust_remote_code modeling file still imports and constructs
cleanly against this image's pinned transformers/timm/einops/kornia versions -- without
paying for a full weight download on every image build, and without ever writing into the
real $HF_HOME (the Dockerfile overrides HF_HOME to a throwaway /tmp dir for just this one
RUN step, deleted immediately after).

CUDA is deliberately never touched here: this build agent has no GPU, and from_config builds
a plain nn.Module on CPU with no .to("cuda") call anywhere in this file -- that call only
happens for real inside handler.py's own module scope, which only runs inside a real GPU job.
"""
import torch

v = torch.__version__
print("torch", v)
assert "cu13" not in v, f"torch was replaced: {v}"

import transformers  # noqa: E402
from transformers import AutoConfig, AutoModelForImageSegmentation  # noqa: E402

print("transformers", transformers.__version__)

for model_id in ("ZhengPeng7/BiRefNet", "ZhengPeng7/BiRefNet_HR", "egeorcun/lucida"):
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageSegmentation.from_config(config, trust_remote_code=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[build_check] {model_id} constructs OK ({n_params:,} params, random init)")

print("[build_check] all three BiRefNet variants construct OK against the pinned stack")
