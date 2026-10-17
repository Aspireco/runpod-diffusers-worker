"""Run at image build time: fail the BUILD, not a billed GPU, when the Wan stack is wrong.

    python /build_check.py pipeline   diffusers has Wan 2.2's MoE second expert
    python /build_check.py lora       a LoRA can go into transformer_2, and peft imports
    python /build_check.py torchao    torchao sits beside diffusers, its fp8 entry point resolves, and peft still
                                      accepts it (peft 0.19 refuses any torchao under 0.16 and then loads no LoRA)

A file rather than `RUN python -c "...\` continuation lines: ACR's dependency scanner reads any line that starts
with `from` as a FROM instruction and fails the build before it begins (run ch19, 2026-10-17).
"""
import inspect
import sys

import torch


def pipeline() -> None:
    import diffusers
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline  # noqa: F401
    p = set(inspect.signature(WanImageToVideoPipeline.__init__).parameters)
    assert "transformer_2" in p, f"diffusers {diffusers.__version__} has no MoE second expert: {sorted(p)}"
    print("diffusers", diffusers.__version__, "| Wan MoE pipeline OK, dual-expert slot present")


def lora() -> None:
    import peft
    from diffusers import WanImageToVideoPipeline
    src = inspect.getsource(WanImageToVideoPipeline.load_lora_weights)
    assert "load_into_transformer_2" in src, "diffusers cannot load a LoRA into transformer_2"
    print("peft", peft.__version__, "| Wan dual-expert LoRA loading OK")


def torchao() -> None:
    import peft
    import peft.import_utils as u
    import torchao as ao
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline  # noqa: F401
    from torchao.quantization import float8_weight_only, quantize_  # noqa: F401
    u.is_torchao_available()      # raises ImportError when peft will not load a LoRA beside this torchao
    print("torchao", ao.__version__, "coexists with diffusers; fp8 arm available | peft", peft.__version__, "accepts it")


if __name__ == "__main__":
    assert "cu13" not in torch.__version__, f"torch replaced: {torch.__version__}"
    {"pipeline": pipeline, "lora": lora, "torchao": torchao}[sys.argv[1]]()
