"""Run at image build time: fail the BUILD, not a billed GPU (a file, because ACR's scanner misreads
Dockerfile RUN lines that begin with `from` -- proof 2026-10-17: `az acr build` of this Dockerfile's
old inline `RUN python -c "... \\nfrom transformers import LlamaModel ..."` line failed its dependency
scan with "unable to understand line from transformers import LlamaModel...", never reaching the
Docker build itself).

Checks torch was not swapped for a cu13 build (Runpod hosts run driver 570/575; CUDA 13 needs >=580
and dies at init before a single log line) and that the `LlamaModel` import chatterbox needs at
runtime -- "Could not import module 'LlamaModel'" against a too-new transformers -- still resolves.
"""
import torch

v = torch.__version__
print("torch", v)
assert "cu13" not in v, f"torch replaced: {v}"

import transformers  # noqa: E402

print("transformers", transformers.__version__)
from transformers import LlamaModel  # noqa: E402,F401

print("LlamaModel import OK")
