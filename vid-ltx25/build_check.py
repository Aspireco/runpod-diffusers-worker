"""Run at image build time: fail the BUILD, not a billed GPU, if torch was swapped or an LTX-2.5 class is missing.

A file rather than `RUN python -c "...\` continuation lines: ACR's dependency scanner reads any line that
starts with `from` as a FROM instruction and fails the build before it begins (run cht, 2026-10-17).
"""
import av
import diffusers
import torch
import transformers
from diffusers import LTX2ConditionPipeline, LTX2Pipeline  # noqa: F401
from diffusers.pipelines.ltx2 import LTX2LatentUpsamplePipeline  # noqa: F401
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel  # noqa: F401
from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition  # noqa: F401
from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES  # noqa: F401
from diffusers.utils import encode_video  # noqa: F401
from transformers.models.gemma4_unified import configuration_gemma4_unified  # noqa: F401

assert "cu13" not in torch.__version__, f"torch replaced: {torch.__version__}"

# handler.UNGUIDED passes these by name: a renamed argument must fail the build, not land in **kwargs on a GPU.
import inspect  # noqa: E402

_params = inspect.signature(LTX2ConditionPipeline.__call__).parameters
_missing = [n for n in ("guidance_scale", "audio_guidance_scale", "stg_scale", "audio_stg_scale", "modality_scale",
                        "audio_modality_scale", "guidance_rescale", "audio_guidance_rescale") if n not in _params]
assert not _missing, f"LTX2ConditionPipeline lacks {_missing}"
print("torch", torch.__version__, "| diffusers", diffusers.__version__, "| transformers", transformers.__version__,
      "| av", av.__version__, "| LTX-2.5 classes OK")
