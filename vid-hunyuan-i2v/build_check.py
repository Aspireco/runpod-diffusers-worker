"""Run at image build time: fail the BUILD, not a billed GPU, if torch was swapped or a
HunyuanVideo-I2V class is missing.

    python /build_check.py imports     every package the handler imports actually imports
    python /build_check.py pipeline    HunyuanVideoImageToVideoPipeline + HunyuanVideoTransformer3DModel
                                        resolve and accept the kwargs handler.py passes them
    python /build_check.py safe_fetch  the SSRF guard refuses non-https, non-public addresses, redirects, oversize/
                                        bodies, and still accepts data: URLs and base64

A file rather than `RUN python -c "..."` continuation lines: ACR's dependency scanner reads any
line that starts with `from` as a FROM instruction and fails the build before it begins
(vid-wan-a14b build_check.py, 2026-10-17).
"""
import inspect
import sys

import torch


def imports() -> None:
    import ftfy  # noqa: F401
    import imageio  # noqa: F401
    import sentencepiece  # noqa: F401
    import transformers  # noqa: F401
    from PIL import Image  # noqa: F401
    print("imports OK: ftfy, imageio, sentencepiece, transformers, PIL")


def pipeline() -> None:
    import diffusers
    from diffusers import HunyuanVideoImageToVideoPipeline
    from diffusers.models import HunyuanVideoTransformer3DModel

    sig = set(inspect.signature(HunyuanVideoImageToVideoPipeline.__call__).parameters)
    for p in ("image", "prompt", "width", "height", "num_frames", "num_inference_steps", "guidance_scale"):
        assert p in sig, f"HunyuanVideoImageToVideoPipeline.__call__ lost {p}: {sorted(sig)}"

    from_pretrained_sig = set(inspect.signature(HunyuanVideoTransformer3DModel.from_pretrained).parameters)
    assert "subfolder" in from_pretrained_sig or "pretrained_model_name_or_path" in from_pretrained_sig, (
        "HunyuanVideoTransformer3DModel.from_pretrained lost its usual diffusers ModelMixin signature"
    )
    print(f"diffusers {diffusers.__version__} | HunyuanVideoImageToVideoPipeline + "
          f"HunyuanVideoTransformer3DModel OK, __call__ has {sorted(p for p in sig if not p.startswith('_'))}")


def safe_fetch_check() -> None:
    """Job inputs can arrive from an LLM agent, so a URL in the input must never reach an
    internal address. safe_fetch.selftest() fakes DNS and the HTTPS connection, so this runs
    with no network on the ACR build agent."""
    import safe_fetch

    safe_fetch.selftest()


if __name__ == "__main__":
    assert "cu13" not in torch.__version__, f"torch replaced: {torch.__version__}"
    {"imports": imports, "pipeline": pipeline, "safe_fetch": safe_fetch_check}[sys.argv[1]]()
