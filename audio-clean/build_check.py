"""Run at image build time: fail the BUILD, not a billed GPU (a file, because ACR's scanner misreads a
`from` line inside a RUN continuation block as a FROM instruction).

ACR build agents have no GPU (see talk-infinitetalk/build_check.py's own note on this) -- both
demucs.api.Separator and df.enhance.init_df() auto-detect CUDA and fall back to CPU on their own, so no
device override is needed here. This goes beyond a bare import: it actually constructs both Demucs
variants against the baked weights and runs init_df() against the baked DeepFilterNet3 checkpoint, so a
corrupt download, a wrong path or the torch.load/weights_only mismatch handler.py works around (see its
own comment) all fail here instead of on the first real job.
"""
import os
from pathlib import Path

import torch

assert "cu13" not in torch.__version__, f"torch replaced: {torch.__version__}"

import runpod  # noqa: E402,F401
import soundfile  # noqa: E402,F401

# Same narrow, restore-immediately-after patch as handler.py -- see its comment for why: demucs's
# load_model() calls torch.load(path, 'cpu') with no override on a checkpoint that pickles an actual
# class reference, not just tensors, which torch >= 2.6's weights_only=True default would reject. Both
# loads below use checkpoints this image's own Dockerfile downloaded and size-checked, not caller data.
_torch_load = torch.load


def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)


torch.load = _torch_load_compat

from demucs.api import Separator  # noqa: E402
from demucs.audio import save_audio  # noqa: E402,F401

DEMUCS_DIR = Path(os.environ.get("DEMUCS_DIR", "/opt/weights/demucs"))
for _name in ("htdemucs", "htdemucs_ft"):
    _sep = Separator(model=_name, repo=DEMUCS_DIR)
    print(f"demucs {_name} loads OK | sr={_sep.samplerate}")

from df.enhance import enhance, init_df  # noqa: E402,F401
from df.io import load_audio, resample  # noqa: E402,F401
from df.io import save_audio as df_save_audio  # noqa: E402,F401
from df.model import ModelParams  # noqa: E402

DFN_DIR = os.environ.get("DFN_DIR", "/opt/weights/DeepFilterNet3")
# 3-tuple (model, df_state, suffix) -- proven by this exact script's own real failure on run ch2f
# (2026-10-17: "expected 4, got 3"), not by the docs, which disagreed with the actual installed source.
_df_model, _df_state, _suffix = init_df(model_base_dir=DFN_DIR, config_allow_defaults=True, log_file=None)
print(f"DeepFilterNet3 loads OK | sr={ModelParams().sr}")

torch.load = _torch_load  # restore torch's own secure default; nothing after this point loads a checkpoint

print("torch", torch.__version__, "| demucs + deepfilternet weights load OK")
