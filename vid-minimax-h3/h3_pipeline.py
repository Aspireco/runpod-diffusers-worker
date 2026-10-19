"""The one place the MiniMax-H3 pipeline is built: handler.py calls build() at boot, and build_check.py calls the same
build() on a decoy directory with the weight loading stubbed, so the exact from_pretrained + load_components sequence
the worker runs is exercised at image build time on CPU.

Why this exists: the first Azure proof (2026-10-19) died at boot with "NotImplementedError: workflows is not supported
because _workflow_map is not set for SequentialPipelineBlocks". from_pretrained(..., workflow="fl2va") already prunes
the blocks to the fl2va workflow's execution blocks (a plain SequentialPipelineBlocks, with no workflow map), and the
handler then passed workflow="fl2va" to load_components too, which calls get_workflow on those pruned blocks
(diffusers 0.41.0 modular_pipeline.py: __init__ line ~1797, load_components line ~2511). The build check had only
called from_pretrained, so it could not see it. Prune once, here.
"""
from __future__ import annotations

import torch
from diffusers import ComponentsManager, ModularPipeline

WORKFLOW = "fl2va"   # text-only t2va runs on the same transformer partition; ref2va needs transformer_ref (not staged)


def build(model_dir: str, *, offload: bool = True, reserve: str = "12GB", device: str = "cuda", verify: bool = True):
    """The pipeline for WORKFLOW from a local flat directory, its components in bf16 in host RAM, and (offload=True)
    a ComponentsManager that moves onto the GPU only what each step needs: diffusers' documented single-80 GB recipe."""
    manager = ComponentsManager()
    pipe = ModularPipeline.from_pretrained(model_dir, workflow=WORKFLOW, components_manager=manager,
                                           local_files_only=True)
    pipe.load_components(dtype=torch.bfloat16)   # NOT workflow=WORKFLOW again: the blocks are already pruned
    if verify:
        # load_components logs a component that failed to load as a WARNING and carries on, so a broken component
        # would only surface mid-job; fail the boot instead, naming it
        missing = [n for n, spec in pipe._component_specs.items()
                   if spec.default_creation_method == "from_pretrained" and getattr(pipe, n, None) is None]
        if missing:
            raise RuntimeError(f"components did not load: {missing} (see the load_components warnings above)")
    if offload:
        manager.enable_auto_cpu_offload(device=device, memory_reserve_margin=reserve)
    return pipe, manager
