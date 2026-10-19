"""Run at image build time: import VibeVoice-ComfyUI's embedded model code the way the node does at its first job, so
a missing dependency fails the BUILD, not a billed GPU (a file, because ACR's dependency scanner misreads a Dockerfile
RUN continuation line that begins with `from`).

The first Azure proof (2026-10-18) died here on a GPU: "No module named 'diffusers'" from vvembed/schedule/dpm_solver.py,
because the node's requirements had not all installed through comfy-node-install.
"""
import sys

NODE = "/comfyui/custom_nodes/VibeVoice-ComfyUI"
sys.path[:0] = [NODE, "/comfyui"]

import diffusers  # noqa: E402
import peft  # noqa: E402
from diffusers import DPMSolverMultistepScheduler  # noqa: E402,F401
from vvembed.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference  # noqa: E402,F401

print(f"vibevoice vvembed imports OK (diffusers {diffusers.__version__}, peft {peft.__version__})")
