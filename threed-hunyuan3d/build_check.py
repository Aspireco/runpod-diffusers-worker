"""Run at image build time: fail the BUILD, not a billed GPU, when the Hunyuan3D-2.1 stack is wrong.

    python /build_check.py imports      every pip package the handler imports actually imports
    python /build_check.py shape        hy3dshape's pipeline + postprocessor API shape resolves
    python /build_check.py paint        hy3dpaint's config + pipeline API shape resolves
    python /build_check.py extensions   custom_rasterizer (CUDA) and mesh_inpaint_processor (C++) import
    python /build_check.py safe_fetch   the SSRF guard refuses http/localhost/loopback/link-local/
                                         private-IP targets and still accepts a data: URL

A file rather than `RUN python -c "..."` continuation lines: ACR's dependency scanner reads any
line that starts with `from` as a FROM instruction and fails the build before it begins
(vid-wan-a14b build_check.py, 2026-10-17). ACR's build agents have no NVIDIA driver at all
(confirmed live on vid-wan-animate, run ch1x, 2026-10-17: a plain `import wan` died with
"Found no NVIDIA driver on your system" because one of its classes evaluates
torch.cuda.current_device() as a default-argument value at class-definition time) -- nothing
below instantiates a model with real weights or launches a CUDA kernel; it only imports modules
and inspects classes, which is why no equivalent patch has been needed here so far. If a future
Hunyuan3D-2.1 commit adds the same pattern somewhere this build hasn't reached yet, the fix is the
same narrow, CPU-build-time-only patch vid-wan-animate's _patch_cuda_current_device_for_cpu_build
uses, added at the exact import that needs it.
"""
import inspect
import sys

import torch

SRC = "/opt/hunyuan3d-src"
for _p in (SRC, f"{SRC}/hy3dshape", f"{SRC}/hy3dpaint"):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def imports() -> None:
    # torchvision_fix (official repo, root level) must run BEFORE `import basicsr`: newer
    # torchvision removed `torchvision.transforms.functional_tensor`, which basicsr's own
    # data/degradations.py still imports unconditionally -- confirmed live, ACR run ch46,
    # 2026-10-18: "ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'".
    # handler.py already calls apply_fix() before any hy3dpaint import for exactly this reason;
    # this check tests imports in isolation from handler.py's own boot sequence, so it needs the
    # same fix applied here too, not inherited from it.
    from torchvision_fix import apply_fix
    apply_fix()

    import basicsr  # noqa: F401
    import bpy  # noqa: F401 - the stub (bpy_stub.py installed as bpy), not real Blender; see
                             # the Dockerfile's "bpy_stub.py is installed AS bpy" comment
    import cv2  # noqa: F401
    import einops  # noqa: F401
    import omegaconf  # noqa: F401
    import pygltflib  # noqa: F401
    import pymeshlab  # noqa: F401
    import realesrgan  # noqa: F401
    import rembg  # noqa: F401
    import skimage  # noqa: F401
    import trimesh  # noqa: F401
    import xatlas  # noqa: F401
    from PIL import Image  # noqa: F401
    print("imports OK: basicsr, bpy, cv2, einops, omegaconf, pygltflib, pymeshlab, realesrgan, "
          "rembg, skimage, trimesh, xatlas, PIL")


def shape() -> None:
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    from hy3dshape.postprocessors import FaceReducer
    from hy3dshape.rembg import BackgroundRemover

    sig = set(inspect.signature(Hunyuan3DDiTFlowMatchingPipeline.__call__).parameters)
    for p in ("image", "num_inference_steps", "guidance_scale", "octree_resolution",
              "num_chunks", "mc_algo", "generator", "output_type"):
        assert p in sig, f"Hunyuan3DDiTFlowMatchingPipeline.__call__ lost {p}: {sorted(sig)}"

    fp_sig = set(inspect.signature(Hunyuan3DDiTFlowMatchingPipeline.from_pretrained).parameters)
    for p in ("model_path", "subfolder", "device", "dtype"):
        assert p in fp_sig, f"Hunyuan3DDiTFlowMatchingPipeline.from_pretrained lost {p}: {sorted(fp_sig)}"

    face_sig = set(inspect.signature(FaceReducer.__call__).parameters)
    assert "max_facenum" in face_sig, f"FaceReducer.__call__ lost max_facenum: {sorted(face_sig)}"

    # BackgroundRemover() constructs a rembg session eagerly (new_session() loads u2net.onnx from
    # disk, no network call once HF_HUB_OFFLINE / the baked-in checkpoint are in place) -- cheap
    # and CPU-only, so this is a real construction, not just a signature check.
    BackgroundRemover()
    print(f"hy3dshape OK: Hunyuan3DDiTFlowMatchingPipeline.__call__ has {sorted(sig)}, "
          f"FaceReducer OK, BackgroundRemover (u2net) constructs on CPU")


def paint() -> None:
    from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline

    conf = Hunyuan3DPaintConfig(6, 512)
    for attr in ("multiview_cfg_path", "custom_pipeline", "multiview_pretrained_path",
                 "dino_ckpt_path", "realesrgan_ckpt_path", "raster_mode"):
        assert hasattr(conf, attr), f"Hunyuan3DPaintConfig lost {attr}"
    assert conf.raster_mode == "cr", f"Hunyuan3DPaintConfig.raster_mode changed from 'cr': {conf.raster_mode}"

    call_sig = set(inspect.signature(Hunyuan3DPaintPipeline.__call__).parameters)
    for p in ("mesh_path", "image_path", "output_mesh_path", "save_glb"):
        assert p in call_sig, f"Hunyuan3DPaintPipeline.__call__ lost {p}: {sorted(call_sig)}"
    print(f"hy3dpaint OK: Hunyuan3DPaintConfig attrs present (raster_mode={conf.raster_mode!r}), "
          f"Hunyuan3DPaintPipeline.__call__ has {sorted(call_sig)}")


def extensions() -> None:
    import custom_rasterizer as cr  # noqa: F401 - the CUDA rasterizer; must import even with no driver
    from DifferentiableRenderer.mesh_inpaint_processor import meshVerticeInpaint  # noqa: F401

    assert hasattr(cr, "rasterize") or hasattr(cr, "__file__"), "custom_rasterizer imported but looks empty"
    print(f"extensions OK: custom_rasterizer ({cr.__file__}), "
          "DifferentiableRenderer.mesh_inpaint_processor.meshVerticeInpaint")


def safe_fetch_check() -> None:
    """Job inputs can arrive from an LLM agent, so a URL in the input must never reach an
    internal address. Checked on literal addresses, not live DNS, so this runs with no network
    on the ACR build agent -- getaddrinfo resolves a literal IP string and "localhost" without
    touching the network either way."""
    import safe_fetch

    for bad in (
        "http://example.com/x.png",
        "https://localhost/x.png",
        "https://127.0.0.1/x.png",
        "https://169.254.169.254/latest/meta-data/",
        "https://10.0.0.1/x.png",
    ):
        try:
            safe_fetch.fetch_image_bytes(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"safe_fetch accepted a target it must refuse: {bad}")

    # a 1x1 PNG -- proves the data: URL path still works, not just that the refusals do
    ok = safe_fetch.fetch_image_bytes(
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
        "+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    assert len(ok) > 0, "safe_fetch failed to decode a valid data: URL"
    print("safe_fetch OK: refuses http/localhost/loopback/link-local/private-IP, accepts data: URLs")


if __name__ == "__main__":
    assert "cu13" not in torch.__version__, f"torch replaced: {torch.__version__}"
    {"imports": imports, "shape": shape, "paint": paint, "extensions": extensions,
     "safe_fetch": safe_fetch_check}[sys.argv[1]]()
