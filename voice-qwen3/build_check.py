"""Run at image build time: fail the BUILD, not a billed GPU (a file, because ACR's scanner misreads `from` lines)."""
import soundfile
import torch
import transformers
from qwen_tts import Qwen3TTSModel  # noqa: F401

assert "cu13" not in torch.__version__, f"torch replaced: {torch.__version__}"
print("torch", torch.__version__, "| transformers", transformers.__version__, "| soundfile", soundfile.__version__,
      "| qwen_tts OK")
