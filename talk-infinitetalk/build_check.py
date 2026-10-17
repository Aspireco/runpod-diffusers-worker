"""Run at image build time: fail the BUILD, not a billed GPU (a file, because ACR's dependency scanner
misreads a `from` line inside a RUN continuation block as a FROM instruction).

Imports everything handler.py imports, at the same sys.path, so a missing/incompatible package or an
upstream rename inside the vendored `wan`/`src` packages is caught here -- before any GPU time is spent.
`import wan` alone is the real test: `wan/__init__.py` pulls in configs, distributed and modules, and
every pipeline class (image2video, text2video, vace, first_last_frame2video, multitalk) exactly as the
official `generate_infinitetalk.py` does, so this single import exercises the package's full dependency
surface -- xformers, xfuser, optimum-quanto, decord, einops, regex, safetensors and the rest.
"""
import sys

import torch

assert "cu13" not in torch.__version__, f"torch replaced: {torch.__version__}"

sys.path.insert(0, "/opt/infinitetalk")

import wan  # noqa: E402,F401
from wan import InfiniteTalkPipeline  # noqa: E402,F401
from wan.configs import WAN_CONFIGS  # noqa: E402

assert "infinitetalk-14B" in WAN_CONFIGS, f"wan.configs lost the infinitetalk-14B task: {sorted(WAN_CONFIGS)}"

from wan.wan_lora import WanLoraWrapper  # noqa: E402,F401

from src.audio_analysis.wav2vec2 import Wav2Vec2Model  # noqa: E402,F401
from transformers import Wav2Vec2FeatureExtractor  # noqa: E402,F401

# The hard, unconditional dependency: wan/modules/attention.py imports xformers.ops with no try/except,
# for the audio cross-attention path (SingleStreamAttention.forward). flash_attn is NOT required -- the
# main attention() function falls back to torch's scaled_dot_product_attention when it is absent.
import xformers  # noqa: E402
import xformers.ops  # noqa: E402,F401
import decord  # noqa: E402,F401
import librosa  # noqa: E402,F401
import soundfile  # noqa: E402,F401
import pyloudnorm  # noqa: E402,F401
import einops  # noqa: E402,F401
import regex  # noqa: E402,F401 -- imported as `re` inside wan/modules/tokenizers.py, not the stdlib `re`
import optimum.quanto  # noqa: E402,F401
from safetensors.torch import load_file  # noqa: E402,F401

import diffusers  # noqa: E402

print("torch", torch.__version__, "| transformers", end=" ")
import transformers  # noqa: E402

print(transformers.__version__, "| diffusers", diffusers.__version__, "| xformers", xformers.__version__,
      "| xfuser", end=" ")
import xfuser  # noqa: E402

print(getattr(xfuser, "__version__", "?"), "| wan package + InfiniteTalkPipeline import OK")
