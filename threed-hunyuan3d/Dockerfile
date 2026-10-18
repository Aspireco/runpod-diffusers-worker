# Hunyuan3D-2.1 (Tencent): image -> textured 3D mesh (shape via Hunyuan3DDiTFlowMatchingPipeline,
# then PBR texture via the hy3dpaint multiview-diffusion paint pipeline). Azure GPU fleet lane
# `hunyuan3d`, A100 80GB PCIe.
#
# LICENCE -- READ README.md FIRST. Tencent Hunyuan 3D 2.1 Community License: Territory is
# worldwide EXCLUDING the European Union, United Kingdom and South Korea, and that exclusion
# binds Outputs (generated meshes) too, not just the weights (LICENSE S5(c)). Outputs may not be
# used to train another AI model (S5(b)). Above 1,000,000 monthly active users, a separate licence
# is required (S4) -- a much lower bar than the 100,000,000 that the Llama-style licences and
# Tencent's OWN video/LLM licences use (see workers/vid-hunyuan-i2v/README.md for that contrast).
# This is exactly why this image is built only in azswmcacr (East US 2, inside the Territory) and
# runs only in the Canada Central GPU environment.
#
# OFFICIAL CODE, PINNED. Everything under /opt/hunyuan3d-src is the official
# Tencent-Hunyuan/Hunyuan3D-2.1 repo at commit 82920d643c0dc2f7bfd7255f45f62d386edfe60c (HEAD of
# `main`, 2026-10-18, latest at build time) -- shape: hy3dshape/hy3dshape/pipelines.py's
# Hunyuan3DDiTFlowMatchingPipeline; texture: hy3dpaint/textureGenPipeline.py's
# Hunyuan3DPaintPipeline (the official multiview-diffusion + Real-ESRGAN + custom_rasterizer PBR
# bake, NOT the ComfyUI core path used by workers/threed/hunyuan/, which is geometry-only -- see
# that folder's own header). handler.py's generation sequence mirrors the official
# model_worker.py's ModelWorker.generate() (api_server.py's actual production code path), not
# just demo.py, for one concrete reason recorded in handler.py and README.md: model_worker.py
# avoids the official `convert_obj_to_glb` Blender export in favour of a pygltflib-based PBR GLB
# builder, which is also what this handler calls.
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV HF_HUB_ENABLE_HF_TRANSFER=0 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    HUNYUAN3D_DIR="/runpod-volume/hunyuan3d" \
    HUNYUAN3D_DINOV2_DIR="/runpod-volume/hunyuan3d-dinov2" \
    TORCH_CUDA_ARCH_LIST="8.0" \
    U2NET_HOME="/opt/hunyuan3d-ckpt/u2net" \
    PYTHONUNBUFFERED=1

# Azure Files (SMB) refuses copystat/chmod, and the mounted share is for pre-staged READ-ONLY
# weights only -- every cache a library might otherwise place under $HOME (which resolves to the
# container's own local disk, not the share, so this is a belt-and-suspenders move, not a fix for
# an observed break) is pinned to /tmp instead, so nothing ever tries to write near the share.
ENV HF_HOME=/tmp/hf_home \
    HF_MODULES_CACHE=/tmp/hf_modules \
    TORCH_HOME=/tmp/torch_home \
    TRITON_CACHE_DIR=/tmp/triton \
    TORCHINDUCTOR_CACHE_DIR=/tmp/inductor \
    XDG_CACHE_HOME=/tmp/xdg_cache

# build-essential/cmake/pkg-config/ninja: compiling custom_rasterizer (CUDAExtension) and
# mesh_inpaint_processor (pybind11 C++, DifferentiableRenderer/compile_mesh_painter.sh).
# python3-dev: python3-config, which compile_mesh_painter.sh calls directly for the extension
# suffix -- sam2's CUDA extension (vid-wan-animate) compiled fine on this same base image WITHOUT
# it via torch.utils.cpp_extension, but that path doesn't shell out to python3-config, so it is
# not proof this one is already present. libeigen3-dev: the official Dockerfile
# (Tencent-Hunyuan/Hunyuan3D-2.1/docker/Dockerfile) installs it immediately before the same
# custom_rasterizer compile step; cheap enough to include rather than find out the hard way that
# grid_neighbor.cpp wants Eigen headers. The libgl1/libegl1/libxrender1/libxi6/libsm6/libxext6
# family: kept defensively even though bpy is stubbed out below (not real Blender, so it no
# longer needs them) -- opencv-python's well-known "ImportError: libGL.so.1" failure needs
# libgl1 regardless, and pymeshlab's bundled Qt can reach for the X11 libs too.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake pkg-config ninja-build python3-dev git \
      libeigen3-dev \
      libgl1 libglx0 libegl1 libgles2 libxrender1 libxi6 libsm6 libxext6 libxxf86vm1 libxkbcommon0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --ignore-installed cryptography
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Pin the base image's torch as constraints (see vid-wan-a14b/Dockerfile for the full story):
# left unpinned, a bare "torch" dependency resolves to a CUDA 13 wheel from PyPI, which needs
# driver >= 580 and crash-loops with an empty log before Python even starts. custom_rasterizer's
# setup.py also needs this SAME torch visible to compile its CUDA extension against (below).
RUN python -c "import torch, torchvision, torchaudio; \
open('/tmp/constraints.txt','w').write(''.join(f'{n}=={v}\n' for n,v in \
[('torch',torch.__version__),('torchvision',torchvision.__version__),('torchaudio',torchaudio.__version__)]))" \
    && cat /tmp/constraints.txt

# Pinned CLOSE to the official requirements.txt (hy3dshape/requirements.txt + the top-level one),
# deliberately tighter than this repo's other lanes: unlike Wan2.2 or LTX-2.5, this code reaches
# into diffusers INTERNALS directly (hy3dshape/hy3dshape/pipelines.py imports
# diffusers.utils.torch_utils.randn_tensor and diffusers.utils.import_utils.is_accelerate_version;
# hy3dpaint/hunyuanpaintpbr/pipeline.py subclasses
# diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline directly
# and imports its free functions retrieve_timesteps/rescale_noise_cfg) rather than only its stable
# public pipeline classes -- version drift here is an AttributeError, not a deprecation warning.
# diffusers/transformers/accelerate are PURE PYTHON with no compiled-extension coupling to torch's
# ABI (unlike torchvision), so an old diffusers on this base image's newer torch 2.8 is a safe
# combination; only custom_rasterizer and mesh_inpaint_processor (below) compile against torch
# and so must build against whatever torch is ACTUALLY installed, which -c constraints.txt holds
# at the base image's 2.8 throughout.
#
# Trimmed from the official requirements.txt: gradio/fastapi/uvicorn (that repo's OWN demo
# servers -- this image has its own handler.py instead), deepspeed/pytorch-lightning/tb_nightly
# (training only, nothing on the inference call path imports them), cupy-cuda12x/open3d/timm/
# pythreejs/torchdiffeq/torchmetrics/pydantic/configargparse (grepped every file on the shape +
# paint call path -- hy3dshape/hy3dshape/{pipelines,postprocessors,preprocessors,rembg,utils}.py,
# hy3dpaint/{textureGenPipeline,convert_utils}.py, hy3dpaint/utils/*.py,
# hy3dpaint/DifferentiableRenderer/{MeshRender,mesh_utils,camera_utils}.py,
# hy3dpaint/hunyuanpaintpbr/{pipeline,unet/modules,unet/attn_processor}.py -- none import them).
# If build_check reveals one of these IS reachable after all, it gets added back with the file
# and line that needed it, not restored wholesale.
# Two of the official pins have no wheel at all for this base image's Python (3.12) and were
# bumped to the nearest compatible release (checked the PyPI JSON API directly, 2026-10-18, not
# assumed from the version number): onnxruntime==1.16.3 tops out at cp311 (1.17.0 is the first
# cp312 build); pymeshlab==2022.2.post3 tops out at cp311 too (2023.12 is the first cp312 build --
# picked .post3, a bugfix point release of that same first cp312-compatible minor version, to
# stay as close to the original pin as the Python constraint allows). Every other exact pin here
# was individually checked the same way and does have a cp312 (or universal py3/py2.py3) wheel.
RUN pip install --no-cache-dir -c /tmp/constraints.txt \
      "ninja==1.11.1.1" "pybind11==2.13.4" \
      "diffusers==0.30.0" "transformers==4.46.0" "accelerate==1.1.1" "huggingface-hub==0.30.2" \
      "safetensors==0.4.4" \
      "numpy>=1.24.4,<2" "scipy==1.14.1" "einops==0.8.0" \
      "opencv-python==4.10.0.84" "imageio==2.36.0" "scikit-image==0.24.0" \
      "rembg==2.0.65" "onnxruntime==1.19.2" \
      "realesrgan==0.3.0" "basicsr==1.4.2" \
      "trimesh==4.4.7" "pymeshlab==2023.12.post3" "pygltflib==1.16.3" "xatlas==0.0.9" \
      "omegaconf==2.3.0" "pyyaml==6.0.2" "tqdm==4.66.5" \
      "runpod>=1.10.1"

# bpy -- NOT pip-installed as the official requirements.txt pins ("bpy==4.0"): that version does
# not exist on PyPI at all, and every real release does (4.2.0-4.5.14: cp311 only; 5.2.2: cp313
# only -- checked the PyPI JSON API directly, 2026-10-18) -- nothing ships a wheel for this base
# image's Python (3.12). mesh_utils.py's TOP-LEVEL `import bpy` still gates the whole module
# (MeshRender.py's own import of it is wrapped in a bare try/except that only prints a warning,
# but MeshRender.save_mesh() unconditionally calls the module-level `save_mesh(...)` function
# that import brings in, NOT behind any save_glb flag -- confirmed by reading
# textureGenPipeline.py's Hunyuan3DPaintPipeline.__call__ directly), but nothing on this
# worker's call path ever touches bpy's actual API: handler.py uses hy3dpaint.convert_utils's
# pygltflib-based GLB builder instead of the Blender-based convert_obj_to_glb (see handler.py's
# module docstring). bpy_stub.py is installed AS bpy instead -- it satisfies the import and
# raises a clear, specific error if anything ever reaches past it into a real Blender operator.
COPY bpy_stub.py /tmp/bpy_stub.py
RUN python -c "import shutil, site; dest = site.getsitepackages()[0] + '/bpy.py'; \
shutil.copy('/tmp/bpy_stub.py', dest); print('bpy stub installed at', dest)"
RUN python -c "import bpy; print('bpy', bpy.app.version_string, 'background =', bpy.app.background)"

# Official repo, pinned to the exact commit this Dockerfile and handler.py were read against.
# Shallow fetch-by-sha (not a branch tip) rather than a full clone: GitHub's smart-HTTP transport
# serves a direct commit fetch for a public repo without the rest of history.
RUN git init /opt/hunyuan3d-src \
 && cd /opt/hunyuan3d-src \
 && git remote add origin https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git \
 && git fetch --depth 1 origin 82920d643c0dc2f7bfd7255f45f62d386edfe60c \
 && git checkout FETCH_HEAD \
 && rm -rf .git

# custom_rasterizer: the CUDA rasterizer MeshRender uses for raster_mode="cr" (the official
# default -- MeshRender.__init__ does a bare `import custom_rasterizer as cr` with no try/except
# when raster_mode=="cr", confirmed by reading DifferentiableRenderer/MeshRender.py directly, so
# this one must succeed). --no-build-isolation: setup.py does
# `from torch.utils.cpp_extension import CUDAExtension` at build time, which needs torch visible
# in THIS environment, not pip's isolated build sandbox -- the same gotcha flash-attn's own
# INSTALL.md documents and vid-wan-animate's sam2 compile already hit. TORCH_CUDA_ARCH_LIST=8.0
# (set above, inherited) targets only the card this runs on (A100, sm_80), not the official
# Dockerfile's full "6.0;6.1;...;9.0" list, matching the task's own instruction and the sm_80-only
# convention vid-wan-animate's sam2 compile already uses.
RUN cd /opt/hunyuan3d-src/hy3dpaint/custom_rasterizer \
 && pip install --no-cache-dir --no-build-isolation -c /tmp/constraints.txt -e .

# mesh_inpaint_processor: the pybind11 C++ (CPU-only, no CUDA) extension
# ViewProcessor.texture_inpaint relies on -- also imported with no try/except at the call site
# that matters (pipeline_utils.py), only MeshRender.py's TOP-of-file import is defensive.
RUN cd /opt/hunyuan3d-src/hy3dpaint/DifferentiableRenderer \
 && bash compile_mesh_painter.sh \
 && python -c "import sys; sys.path.insert(0, '.'); import mesh_inpaint_processor; print('mesh_inpaint_processor', mesh_inpaint_processor.__file__)"

# Real-ESRGAN x4plus checkpoint (BSD-3-Clause, xinntao/Real-ESRGAN -- confirmed against that
# repo's own LICENSE file, 2026-10-18): imageSuperNet's upscale pass for the baked albedo/MR
# textures. ~64MB -- small enough to bake into the image rather than fetch at runtime, same
# allowance vid-wan-animate used for its ~170MB detector checkpoint. Official download URL and
# destination path, from the official Dockerfile and textureGenPipeline.py's own default.
RUN mkdir -p /opt/hunyuan3d-src/hy3dpaint/ckpt \
 && python -c "import urllib.request; urllib.request.urlretrieve( \
      'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth', \
      '/opt/hunyuan3d-src/hy3dpaint/ckpt/RealESRGAN_x4plus.pth')" \
 && python -c "import os; n='/opt/hunyuan3d-src/hy3dpaint/ckpt/RealESRGAN_x4plus.pth'; \
sz=os.path.getsize(n); assert sz > 60_000_000, f'RealESRGAN checkpoint too small: {sz} bytes'; print('RealESRGAN_x4plus.pth', sz, 'bytes')"

# u2net.onnx (Apache-2.0, NathanUA/U-2-Net -- rembg's default background-removal model, used by
# hy3dshape.rembg.BackgroundRemover's new_session() with no model_name override, i.e. "u2net").
# Baked in at the EXACT path the installed rembg version's own session class resolves, discovered
# by asking that class directly rather than hardcoding a path that drifts across rembg releases.
# A file, not a RUN python -c "..." block: this needs a `from rembg.sessions.u2net import ...`
# line, and ACR's dependency scanner reads any line starting with `from` as a FROM instruction and
# fails the build before it begins -- caught live on this exact line, ACR run ch3x, 2026-10-18.
COPY bake_u2net.py /bake_u2net.py
RUN python /bake_u2net.py

# safe_fetch.py: the SSRF guard for any job input that is an https:// URL rather than inline data
# (job inputs can arrive from an LLM agent, not just a human -- see its own module docstring).
# Copied in before build_check.py so build_check's own safe_fetch subcommand can import it.
COPY safe_fetch.py /safe_fetch.py

# Fail the BUILD, not a billed GPU. build_check.py imports everything the handler imports and
# constructs whatever can be constructed on CPU.
COPY build_check.py /build_check.py
RUN python -c "import torch; v=torch.__version__; print('torch', v); assert 'cu13' not in v, f'torch replaced: {v}'"
RUN python /build_check.py imports
RUN python /build_check.py shape
RUN python /build_check.py paint
RUN python /build_check.py extensions
RUN python /build_check.py safe_fetch

COPY handler.py /handler.py
RUN python -c "import ast; ast.parse(open('/handler.py').read()); print('handler parses OK')"

CMD ["python", "-u", "/handler.py"]
