"""Run at image build time: fail the BUILD, not a billed GPU, if torch was swapped, a MiniMax-H3 class is
missing, or the offline-loading trap below silently regressed in a future diffusers bump.

    python /build_check.py imports     every package handler.py imports actually imports, including the
                                        minimax_h3 module and its reference classes, and the Qwen3-VL
                                        conditioner class
    python /build_check.py offline     proves the local-directory loading trick handler.py relies on, on a
                                        small decoy directory -- no GPU, no real 144GB of weights
    python /build_check.py safe_fetch  the SSRF guard refuses non-https, non-public addresses, redirects,
                                        oversize bodies, and still accepts data: URLs and base64

A file rather than `RUN python -c "..."` continuation lines: ACR's dependency scanner reads any line that
starts with `from` as a FROM instruction and fails the build before it begins (vid-wan-a14b build_check.py,
2026-10-17).
"""
import sys
import tempfile

import torch


def imports() -> None:
    import av  # noqa: F401
    import diffusers
    import transformers
    from diffusers import ComponentsManager, ModularPipeline  # noqa: F401
    from diffusers.modular_pipelines.minimax_h3 import (  # noqa: F401
        MiniMaxH3AudioReference,
        MiniMaxH3ImageReference,
        MiniMaxH3VideoReference,
    )
    from diffusers.utils.export_utils import encode_video  # noqa: F401
    from transformers import Qwen3VLForConditionalGeneration  # noqa: F401
    print(f"diffusers {diffusers.__version__} | transformers {transformers.__version__} | "
          f"av {av.__version__} | ModularPipeline, ComponentsManager, MiniMaxH3*Reference, "
          f"Qwen3VLForConditionalGeneration, encode_video all import OK")


def offline() -> None:
    """Proves handler.py's offline-loading trick on a decoy directory, without the real 144GB of weights.

    `modular_model_index.json` names every component with `pretrained_model_name_or_path:
    "MiniMaxAI/MiniMax-H3"`. diffusers' own `ModularPipeline.from_pretrained` rewrites a component's spec
    to the LOCAL path it was called with the instant `<path>/<subfolder>` already contains that
    component's files (`_is_local_component` in diffusers/modular_pipelines/modular_pipeline.py: a
    weight-named file for a model component, the scheduler's own config filename for a scheduler, one of
    `TRANSFORMERS_COMPONENT_AUX_FILES` for a tokenizer/processor). That check only reads `os.listdir` and
    filename suffixes -- it never opens a file -- so a zero-byte file with the right name is exactly as
    convincing to it as a real 61.7GB shard, which is what lets this run on CPU with no network at build
    time and still prove the real mechanism, not a mock of it.

    Downloads only `model_index.json`, `modular_model_index.json` and the handful of small per-component
    config/tokenizer/processor/scheduler files the repo carries (combined, a few MB) -- never a weight
    file, which would make a CPU build agent pull multiple GB for a check that does not need them.
    """
    from huggingface_hub import hf_hub_download

    repo = "MiniMaxAI/MiniMax-H3"
    # Every file `_is_local_component` needs to see for a non-model component (scheduler config name /
    # a TRANSFORMERS_COMPONENT_AUX_FILES match), plus one config.json per model component for realism
    # (not required by the check itself, but cheap and makes the decoy dir a truer stand-in).
    small_files = [
        "model_index.json",
        "modular_model_index.json",
        "transformer/config.json",
        "text_encoder/config.json",
        "vae/config.json",
        "audio_vae/config.json",
        "scheduler/scheduler_config.json",
        "audio_scheduler/scheduler_config.json",
        "tokenizer/tokenizer_config.json",
        "processor/preprocessor_config.json",
    ]
    # filter_model_files (diffusers/pipelines/pipeline_loading_utils.py) only checks a filename's
    # extension, never its content -- so an empty file with a weight extension is indistinguishable from
    # a real shard to `_is_local_component`, and the real weights are never fetched by this check.
    weight_placeholders = [
        "transformer/dummy-weights.safetensors",
        "text_encoder/dummy-weights.safetensors",
        "vae/dummy-weights.safetensors",
        "audio_vae/dummy-weights.safetensors",
    ]

    with tempfile.TemporaryDirectory() as cache_dir, tempfile.TemporaryDirectory() as decoy_dir:
        import os
        import shutil

        for rel in small_files:
            fetched = hf_hub_download(repo, rel, cache_dir=cache_dir)
            dest = os.path.join(decoy_dir, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy(fetched, dest)
        for rel in weight_placeholders:
            dest = os.path.join(decoy_dir, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            open(dest, "wb").close()

        # The handler's own build(), with only the per-component weight load stubbed (the placeholders are empty):
        # from_pretrained + load_components run exactly as at boot, which is where the first Azure proof died
        # (load_components(workflow=...) on blocks from_pretrained had already pruned, 2026-10-19).
        from diffusers.modular_pipelines.modular_pipeline_utils import ComponentSpec

        import h3_pipeline

        loaded = []
        real_load = ComponentSpec.load
        ComponentSpec.load = lambda spec, **kw: loaded.append(spec.name)
        try:
            pipe, _ = h3_pipeline.build(decoy_dir, offload=False, verify=False)
        finally:
            ComponentSpec.load = real_load
        assert loaded, "load_components loaded nothing: the workflow pruning or the diffusers version changed"
        assert "transformer" in loaded and "transformer_ref" not in loaded, f"load_components loaded {loaded}"
        print(f"build OK: h3_pipeline.build() ran from_pretrained + load_components; would load {sorted(loaded)}")

        checked = 0
        for name, spec in pipe._component_specs.items():
            if spec.default_creation_method != "from_pretrained":
                continue  # e.g. image_processor/video_processor are from_config, never hit the Hub
            assert spec.pretrained_model_name_or_path == decoy_dir, (
                f"component '{name}' did not resolve to the local dir: "
                f"pretrained_model_name_or_path={spec.pretrained_model_name_or_path!r} "
                f"(still pointing at the Hub would mean a real worker calls out at boot -- see this "
                f"file's module docstring)"
            )
            checked += 1
        assert checked > 0, "no from_pretrained component specs found at all -- the workflow selection or " \
                             "the diffusers version regressed"
        # transformer_ref is the ref2va workflow's own partition; this lane only stages transformer/, so
        # it must be absent once the fl2va workflow has pruned the blocks, not merely unresolved.
        assert "transformer_ref" not in pipe._component_specs, (
            "the fl2va workflow pulled in transformer_ref -- this lane's manifest and fetch job only "
            "stage the transformer/ partition, see README.md Premade search"
        )
        print(f"offline OK: {checked} from_pretrained component(s) resolved to the local dir "
              f"{list(n for n, s in pipe._component_specs.items() if s.default_creation_method == 'from_pretrained')}")


def safe_fetch_check() -> None:
    """Job inputs can arrive from an LLM agent, so a URL in the input must never reach an internal
    address. safe_fetch.selftest() fakes DNS and the HTTPS connection, so this runs with no network on
    the ACR build agent."""
    import safe_fetch

    safe_fetch.selftest()


if __name__ == "__main__":
    assert "cu13" not in torch.__version__, f"torch replaced: {torch.__version__}"
    {"imports": imports, "offline": offline, "safe_fetch": safe_fetch_check}[sys.argv[1]]()
