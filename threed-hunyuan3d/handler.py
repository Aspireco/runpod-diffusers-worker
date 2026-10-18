"""Hunyuan3D-2.1 (Tencent): image -> textured 3D mesh, for the Azure GPU fleet lane `hunyuan3d`
(A100 80GB PCIe).

Uses the OFFICIAL Tencent-Hunyuan/Hunyuan3D-2.1 code directly (vendored at /opt/hunyuan3d-src,
pinned commit -- see Dockerfile): shape generation through
`hy3dshape.pipelines.Hunyuan3DDiTFlowMatchingPipeline` (the diffusion/flow-matching DiT + VAE),
then PBR texturing through `hy3dpaint.textureGenPipeline.Hunyuan3DPaintPipeline` (multiview
diffusion + Real-ESRGAN super-res + the custom_rasterizer CUDA bake). This is the SAME call
sequence as the official repo's own production `model_worker.py` (api_server.py's actual code
path), not just its bare `demo.py` -- see "Why model_worker.py, not demo.py" below for the one
concrete reason that distinction matters here.

LICENCE -- READ BEFORE PRODUCTION USE. Tencent Hunyuan 3D 2.1 Community License. Territory is
worldwide **excluding the European Union, United Kingdom and South Korea** (LICENSE S1(l)), and
that exclusion binds *Outputs* (generated meshes), not just the weights: S5(c) -- "You must not
use, reproduce, modify, distribute, or display the Tencent Hunyuan 3D 2.1 Works, Output or
results ... outside the Territory." S5(b) separately forbids using Outputs to train any other AI
model. S4's monthly-active-user threshold is **1,000,000** -- two orders of magnitude lower than
the 100,000,000 the Tencent Hunyuan VIDEO licence (see workers/vid-hunyuan-i2v) and the
Llama-style licences use. Full table in README.md. This is why the image is built only in
azswmcacr (East US 2, inside the Territory) and runs only in the Canada Central GPU environment.

Why model_worker.py, not demo.py, for the generation sequence
---------------------------------------------------------------
Both of the official repo's own entry points background-remove and shape-generate the same way.
Where they diverge is the final OBJ -> GLB conversion: `textureGenPipeline.Hunyuan3DPaintPipeline
.__call__(..., save_glb=True)` (demo.py's path) calls `DifferentiableRenderer.mesh_utils
.convert_obj_to_glb`, which drives headless Blender (`bpy`) through `bpy.ops.wm.obj_import` /
`bpy.ops.export_scene.gltf` -- an OBJ MTL file has no standard PBR metallic/roughness
representation, so this conversion's fidelity for the metallic/roughness maps depends on
Blender's own OBJ importer, which is not guaranteed to round-trip `map_Pm`/`map_Pr` into glTF's
`pbrMetallicRoughness` correctly. `model_worker.py` (the code actually behind `api_server.py`'s
`/generate` endpoint) instead calls the paint pipeline with `save_glb=False` and does its own
conversion via `hy3dpaint.convert_utils.create_glb_with_pbr_materials` -- a `pygltflib`-based
builder that constructs the glTF PBR material graph explicitly from the separate albedo/
metallic/roughness image files. That is the more deliberate, production-facing choice (it is
literally what Tencent's own API server ships), so this handler follows it, via
`_convert_to_glb_pbr()` below. `bpy` is still a real import-time dependency regardless (see
Dockerfile's "bpy is a real dependency" comment) because `MeshRender.save_mesh()` -- called
unconditionally inside the paint pipeline, not gated by `save_glb` -- delegates to a module-level
`save_mesh()` function that only exists if `DifferentiableRenderer/mesh_utils.py` (which starts
with a bare `import bpy`) imported successfully. This handler just never calls the ONE function
in that module that actually drives Blender's operators.

A second, independent fix from the official code: both `demo.py` and `model_worker.py` write
`image = image.convert("RGBA")` and THEN check `if image.mode == "RGB":` before running
background removal -- a check that can never be true, because the line above just forced the
mode to "RGBA". Confirmed by reading both files directly, not assumed; both ship the identical
ordering mistake, so this is almost certainly an upstream bug rather than an intentional
no-op. Background removal has never actually run through either official entry point as a
result. Fixed here by checking `remove_background` directly against the job input instead of
reconstructing a mode check that cannot fire -- see `handler()` below.
"""

import base64
import io
import os
import sys
import tempfile
import time
import urllib.request

print("[boot] hunyuan3d handler starting", flush=True)

import torch  # noqa: E402

print(f"[boot] torch {torch.__version__} cuda={torch.cuda.is_available()}", flush=True)

SRC = "/opt/hunyuan3d-src"
HUNYUAN3D_DIR = os.environ.get("HUNYUAN3D_DIR", "/runpod-volume/hunyuan3d")
HUNYUAN3D_DINOV2_DIR = os.environ.get("HUNYUAN3D_DINOV2_DIR", "/runpod-volume/hunyuan3d-dinov2")

# hy3dshape/hy3dpaint are used by adding their folders to sys.path and importing the inner
# package / loose modules, exactly as the official demo.py and model_worker.py do -- neither
# ships a setup.py, so there is no `pip install` for them (only custom_rasterizer does, see
# Dockerfile). SRC itself is also added for the repo-root `torchvision_fix.py` module that both
# official entry points import before anything else.
for _p in (SRC, f"{SRC}/hy3dshape", f"{SRC}/hy3dpaint"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from torchvision_fix import apply_fix
    apply_fix()
except Exception as e:  # noqa: BLE001 - matches the official entry points' own non-fatal handling
    print(f"[boot] torchvision_fix not applied: {e}", flush=True)

# One representative file per component the handler actually loads, not all 25 manifest entries
# across the two manifests (gpu/manifests/hunyuan3d.tsv + hunyuan3d-dinov2.tsv list the exact
# set; this is a fast boot check, not a full integrity scan -- same pattern as vid-wan-animate).
_REQUIRED = [
    (HUNYUAN3D_DIR, "hunyuan3d-dit-v2-1/config.yaml"),
    (HUNYUAN3D_DIR, "hunyuan3d-dit-v2-1/model.fp16.ckpt"),
    (HUNYUAN3D_DIR, "hunyuan3d-paintpbr-v2-1/model_index.json"),
    (HUNYUAN3D_DIR, "hunyuan3d-paintpbr-v2-1/unet/diffusion_pytorch_model.bin"),
    (HUNYUAN3D_DIR, "hunyuan3d-paintpbr-v2-1/text_encoder/pytorch_model.bin"),
    (HUNYUAN3D_DINOV2_DIR, "model.safetensors"),
]
for _base, _rel in _REQUIRED:
    _full = os.path.join(_base, _rel)
    if not os.path.isfile(_full):
        raise RuntimeError(
            f"{_full} is missing: run the fetch job for lane hunyuan3d (gpu/manifests/hunyuan3d.tsv "
            f"and gpu/manifests/hunyuan3d-dinov2.tsv) before starting this worker."
        )
print(f"[boot] weights present at {HUNYUAN3D_DIR} and {HUNYUAN3D_DINOV2_DIR}", flush=True)

# ---- the one offline-loading fix: hy3dshape/hy3dpaint call huggingface_hub.snapshot_download()
# with a bare repo id (`tencent/Hunyuan3D-2.1`) and no local_dir/cache_dir override, from two call
# sites -- hy3dshape's smart_load_model() (its own local-path check misses because HY3DGEN_MODELS
# is left at its unset default, so it always falls through) and hy3dpaint's
# multiviewDiffusionNet.__init__(). With HF_HUB_OFFLINE=1 that would need a real Hub cache tree
# (blobs + snapshots, with symlinks) to resolve -- fragile over Azure Files SMB, which every other
# lane's fetch job avoids by using a flat `local_dir` layout instead (see gpu/fetch/fetch_xet.py).
# This worker has exactly one caller-visible repo id, so redirect both call sites to the flat
# mirror unconditionally rather than reconstructing the Hub cache format -- same pattern as
# vid-wan-animate's `pose2d.Yolo = detector_adapter.PersonDetector` monkey-patch, applied to
# huggingface_hub.snapshot_download instead of a vendored class attribute.
import huggingface_hub  # noqa: E402

_real_snapshot_download = huggingface_hub.snapshot_download


def _offline_snapshot_download(*args, repo_id=None, **kwargs):
    if repo_id and "hunyuan3d-2" in repo_id.lower():
        return HUNYUAN3D_DIR
    return _real_snapshot_download(*args, repo_id=repo_id, **kwargs)  # pragma: no cover - not hit in this worker


huggingface_hub.snapshot_download = _offline_snapshot_download

from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline  # noqa: E402
from hy3dshape.postprocessors import FaceReducer  # noqa: E402
from hy3dshape.rembg import BackgroundRemover  # noqa: E402
from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline  # noqa: E402
from convert_utils import create_glb_with_pbr_materials  # noqa: E402

_t0 = time.time()
print("[boot] loading shape pipeline (Hunyuan3DDiTFlowMatchingPipeline)...", flush=True)
# model_path is the literal repo id string the official demo.py/model_worker.py pass; our
# snapshot_download patch above redirects it to HUNYUAN3D_DIR. All other args (device='cuda',
# dtype=torch.float16, variant='fp16', use_safetensors=False, subfolder='hunyuan3d-dit-v2-1') are
# the library's own defaults, matching what model.fp16.ckpt actually ships (see manifest).
SHAPE_PIPE = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2.1")
REDUCE_FACES = FaceReducer()
REMBG = BackgroundRemover()
print(f"[boot] shape pipeline + rembg ready in {time.time() - _t0:.1f}s", flush=True)

print("[boot] loading paint pipeline (Hunyuan3DPaintPipeline: multiview diffusion + "
      "Real-ESRGAN + custom_rasterizer)...", flush=True)
_PAINT_CFG = Hunyuan3DPaintConfig(max_num_view=6, resolution=512)
_PAINT_CFG.realesrgan_ckpt_path = f"{SRC}/hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
_PAINT_CFG.multiview_cfg_path = f"{SRC}/hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
_PAINT_CFG.custom_pipeline = f"{SRC}/hy3dpaint/hunyuanpaintpbr"
_PAINT_CFG.dino_ckpt_path = HUNYUAN3D_DINOV2_DIR  # a local dir, not a repo id -- transformers'
                                                   # from_pretrained takes that path offline with
                                                   # no patch needed (unlike the two calls above).
PAINT_PIPE = Hunyuan3DPaintPipeline(_PAINT_CFG)
print(f"[boot] ready in {time.time() - _t0:.1f}s total", flush=True)


def _load_image(job_input):
    src = job_input.get("image") or job_input.get("image_url") or job_input.get("image_base64")
    if not src:
        return None
    from PIL import Image
    if src.startswith("http"):
        with urllib.request.urlopen(src, timeout=90) as r:
            data = r.read()
    else:
        raw = src.split(",", 1)[1] if src.startswith("data:") else src
        data = base64.b64decode(raw, validate=True)
    return Image.open(io.BytesIO(data))


def _convert_to_glb_pbr(obj_path: str, glb_path: str) -> None:
    """The model_worker.py-style conversion (see module docstring): pygltflib, no Blender."""
    textures = {
        "albedo": obj_path.replace(".obj", ".jpg"),
        "metallic": obj_path.replace(".obj", "_metallic.jpg"),
        "roughness": obj_path.replace(".obj", "_roughness.jpg"),
    }
    textures = {k: v for k, v in textures.items() if os.path.isfile(v)}
    create_glb_with_pbr_materials(obj_path, textures, glb_path)


def handler(job):
    job_input = job.get("input") or {}
    image = _load_image(job_input)
    if image is None:
        return {"error": "image is required (data URL, raw base64, or an https:// URL)"}

    remove_background = bool(job_input.get("remove_background", True))
    texture = bool(job_input.get("texture", True))
    target_faces = int(job_input.get("target_faces", 40000))
    steps = int(job_input.get("steps", 50))
    octree_resolution = int(job_input.get("octree_resolution", 384))
    guidance_scale = float(job_input.get("guidance_scale", 5.0))
    seed = job_input.get("seed", 1234)
    seed = int(seed) if seed is not None else int.from_bytes(os.urandom(4), "big")

    started = time.time()
    work = tempfile.mkdtemp(prefix="hunyuan3d_")
    try:
        if remove_background:
            # See module docstring: the official demo.py/model_worker.py check `image.mode ==
            # "RGB"` AFTER already force-converting to RGBA, which can never be true, so
            # background removal never actually runs through either of them. Honour the job's
            # own flag directly instead.
            image = REMBG(image.convert("RGB"))
        else:
            image = image.convert("RGBA")

        generator = torch.Generator(device="cpu").manual_seed(seed)
        gen_t0 = time.time()
        mesh = SHAPE_PIPE(
            image=image,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            octree_resolution=octree_resolution,
            generator=generator,
        )[0]
        print(f"[run] shape generation done in {time.time() - gen_t0:.1f}s "
              f"({len(mesh.faces)} faces before reduction)", flush=True)

        if target_faces and len(mesh.faces) > target_faces:
            mesh = REDUCE_FACES(mesh, max_facenum=target_faces)
            print(f"[run] FaceReducer -> {len(mesh.faces)} faces (target {target_faces})", flush=True)

        initial_glb = os.path.join(work, "shape.glb")
        mesh.export(initial_glb)

        textured = False
        paint_seconds = 0.0
        final_glb = initial_glb
        if texture:
            paint_t0 = time.time()
            try:
                obj_path = os.path.join(work, "textured.obj")
                # save_glb=False: see module docstring for why this follows model_worker.py's
                # pygltflib conversion instead of the bpy-based one the pipeline would otherwise
                # run when save_glb=True. use_remesh defaults to True inside this call (the
                # official pipeline's own default) and re-simplifies to its own hardcoded 40000
                # faces regardless of target_faces above -- see README's "known deviations".
                obj_path = PAINT_PIPE(
                    mesh_path=initial_glb,
                    image_path=image,
                    output_mesh_path=obj_path,
                    save_glb=False,
                )
                final_glb = os.path.join(work, "textured.glb")
                _convert_to_glb_pbr(obj_path, final_glb)
                textured = True
                paint_seconds = time.time() - paint_t0
                print(f"[run] texture generation done in {paint_seconds:.1f}s", flush=True)
            except Exception as e:  # noqa: BLE001 - fall back to the untextured mesh, same as
                                     # model_worker.py's own try/except around texturing
                print(f"[run] texture generation failed, returning untextured mesh: "
                      f"{type(e).__name__}: {e}", flush=True)
                final_glb = initial_glb
                paint_seconds = time.time() - paint_t0

        import trimesh
        final_mesh = trimesh.load(final_glb, force="mesh")
        faces = int(len(final_mesh.faces))
        vertices = int(len(final_mesh.vertices))

        with open(final_glb, "rb") as f:
            glb_bytes = f.read()

        return {
            "meshes": [{
                "filename": os.path.basename(final_glb),
                "bytes": len(glb_bytes),
                "data": "data:model/gltf-binary;base64," + base64.b64encode(glb_bytes).decode(),
            }],
            "all_files": [os.path.basename(final_glb)],
            "faces": faces,
            "vertices": vertices,
            "textured": textured,
            "seed": seed,
            "steps": steps,
            "octree_resolution": octree_resolution,
            "generation_seconds": round(time.time() - started, 2),
            "paint_seconds": round(paint_seconds, 2),
            "model": "tencent/Hunyuan3D-2.1",
            "licence": "Tencent Hunyuan 3D 2.1 Community (Territory excludes EU/UK/South Korea; "
                       "binds Outputs; 1M MAU threshold)",
        }
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        return {"error": f"CUDA OOM: {e}. Lower octree_resolution or disable texture."}
    except Exception as e:  # surface the reason rather than dying silently
        return {"error": f"{type(e).__name__}: {e}"}
    finally:
        import shutil
        shutil.rmtree(work, ignore_errors=True)
        print(f"[run] job finished in {time.time() - started:.1f}s total", flush=True)


import runpod  # noqa: E402

runpod.serverless.start({"handler": handler})
