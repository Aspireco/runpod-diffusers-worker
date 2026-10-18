"""Run at image build time: fail the BUILD, not a billed GPU (a file, because ACR's scanner
misreads a Dockerfile RUN continuation line that begins with `from` as a FROM instruction).

Two things are checked:

1. Every diffusers/transformers/huggingface_hub/safetensors class handler.py imports still
   resolves at the pinned versions -- an import, nothing more, same as every other
   build_check.py in this repo (e.g. lipsync-musetalk's own).
2. The conv_in-widening tensor surgery inside handler.py's `_build_iclight_unet` -- the one
   piece of this worker that is not a stock diffusers call -- against a throwaway
   UNet2DConditionModel built from a tiny hand-written config, never `from_pretrained`.

This is deliberately NOT a call into handler.py itself: handler.py's module scope hard-codes
`DEVICE = torch.device("cuda")` with no CPU fallback (this build agent has no GPU) and
downloads ~5.6GB of real weights the instant it is imported -- both wrong for a check that
should be cheap and fast on every build. The widening logic below is copied by hand from
handler.py's `_build_iclight_unet` (see its own module docstring, "HOW IC-LIGHT ACTUALLY
WORKS") -- if that function's widening step ever changes, update this copy to match.
"""
import torch
from diffusers import (  # noqa: E402,F401
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.models.attention_processor import AttnProcessor2_0  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402,F401
from safetensors.torch import load_file as load_safetensors  # noqa: E402,F401
from transformers import CLIPTextModel, CLIPTokenizer  # noqa: E402,F401

v = torch.__version__
print("torch", v)
assert "cu13" not in v, f"torch was replaced: {v}"


def _widen_conv_in(unet, in_channels):
    """Mirrors handler.py's _build_iclight_unet widening step -- see this file's own
    docstring for why the real function cannot just be imported here instead."""
    new_conv_in = torch.nn.Conv2d(
        in_channels,
        unet.conv_in.out_channels,
        unet.conv_in.kernel_size,
        unet.conv_in.stride,
        unet.conv_in.padding,
    )
    with torch.no_grad():
        new_conv_in.weight.zero_()
        new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
        new_conv_in.bias = unet.conv_in.bias
    return new_conv_in


# A real SD1.5 UNet2DConditionModel's conv_in is Conv2d(4, 320, kernel_size=3, padding=1).
# This tiny stand-in keeps conv_in at 4 input channels so the widening code sees the same
# starting shape it sees in production, without downloading or holding the real ~1.7GB UNet.
tiny = UNet2DConditionModel(
    sample_size=8, in_channels=4, out_channels=4, layers_per_block=1,
    block_out_channels=(32,), down_block_types=("DownBlock2D",),
    up_block_types=("UpBlock2D",), cross_attention_dim=8,
)
assert tiny.conv_in.in_channels == 4

for extra_channels, label in ((12, "fbc"), (8, "fc")):
    widened = _widen_conv_in(tiny, extra_channels)
    assert widened.in_channels == extra_channels
    assert widened.out_channels == tiny.conv_in.out_channels
    assert torch.equal(widened.weight[:, :4], tiny.conv_in.weight), "pretrained 4ch not preserved"
    assert widened.weight[:, 4:].abs().max().item() == 0.0, "new channels were not zeroed"
    assert widened.bias is tiny.conv_in.bias, "bias should be shared, not copied"
    print(f"[build_check] conv_in widen 4->{extra_channels} ({label}) OK")

DPMSolverMultistepScheduler(
    num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
    algorithm_type="sde-dpmsolver++", use_karras_sigmas=True, steps_offset=1,
)
AttnProcessor2_0()
print("[build_check] diffusers/transformers/huggingface_hub/safetensors imports + "
      "conv_in widening construct OK")
