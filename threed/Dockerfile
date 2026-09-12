# 3D generation lane: product photo or room photo -> textured mesh.
#
# WHY COMFYUI AND NOT A HAND-BUILT WORKER. TRELLIS.2's own install is a
# `git clone --recursive` plus a setup.sh that compiles flash-attn, nvdiffrast, a
# custom sparse-conv kernel and diff-gaussian-rasterization against whatever CUDA it
# finds -- exactly the shape of the day that was lost resolving dependencies from a
# bare PyTorch base. ComfyUI 0.34.0 ships TRELLIS.2 in *core*: comfy/ldm/trellis2 and
# comfy_extras/nodes_trellis2.py. There is no custom node to install and no kernel to
# compile. That is the whole reason this lane is cheap.
#
# VERSION COUPLING, LOAD-BEARING. runpod/worker-comfyui:5.10.0-base pins ComfyUI
# 0.34.0. Every node the vendor template uses was verified present at that tag --
# including the three that live in files easy to miss: RemoveBackground and
# LoadBackgroundRemovalModel in nodes_bg_removal.py, CFGOverride in nodes_cfg.py.
# Bumping the base image without re-running build_api_workflows.py is how this breaks.
#
# LICENCE POSITION (full working in stack-docs/3D-BENCHMARK.md):
#   TRELLIS.2          MIT (microsoft/TRELLIS.2, weights included)     -- clean
#   MoGe-2             MIT (Comfy-Org/MoGe <- Ruicheng/moge-2-vitl)    -- clean
#   BiRefNet           MIT (Comfy-Org/BiRefNet)                        -- clean
#   DINOv3 encoder     Meta DINOv3 Licence -- royalty-free, worldwide, commercial use
#                      permitted, no revenue or MAU cap. NOT MIT despite the Comfy-Org
#                      repo tag; see DECISIONS.md Part 3.
# Hunyuan3D 2.1 is deliberately NOT in this image. It is quarantined in hunyuan/ --
# its licence excludes the EU, UK and South Korea for the *outputs* too.
FROM runpod/worker-comfyui:5.10.0-base

# No custom nodes. Everything below is core ComfyUI.
#
# int8_convrot over bf16: 5.25GB vs 10.34GB for output the benchmark could not tell
# apart on product shots, and it is the difference between a cold start that pulls
# 9GB and one that pulls 14GB. bf16 is a one-line change if a client ever needs it.
RUN comfy model download \
      --url https://huggingface.co/Comfy-Org/TRELLIS.2/resolve/main/diffusion_models/trellis_2_int8_convrot.safetensors \
      --relative-path models/diffusion_models \
      --filename trellis_2_int8_convrot.safetensors

RUN comfy model download \
      --url https://huggingface.co/Comfy-Org/TRELLIS.2/resolve/main/vae/trellis_2_shape_vae_bf16.safetensors \
      --relative-path models/vae \
      --filename trellis_2_shape_vae_bf16.safetensors \
 && comfy model download \
      --url https://huggingface.co/Comfy-Org/TRELLIS.2/resolve/main/vae/trellis_2_texture_vae_bf16.safetensors \
      --relative-path models/vae \
      --filename trellis_2_texture_vae_bf16.safetensors

# The image encoder. Filename matters: the vendor template's CLIPVisionLoader asks for
# dino_v3_L_naf_fp32.safetensors, which lives in Comfy-Org/Pixal3D, NOT in
# Comfy-Org/TRELLIS.2 (that repo ships a differently-named dino_v3_vit_l.safetensors).
# Pulling the TRELLIS.2 one and renaming it produces a graph that loads and generates
# noise, because they are not the same tensor layout.
RUN comfy model download \
      --url https://huggingface.co/Comfy-Org/Pixal3D/resolve/main/clip_vision/dino_v3_L_naf_fp32.safetensors \
      --relative-path models/clip_vision \
      --filename dino_v3_L_naf_fp32.safetensors

# MoGe-2 estimates per-pixel geometry and camera FOV from one photo. In the TRELLIS.2
# graph it supplies the FOV that sets object proportions. On its own it is the
# room-scale lane: TRELLIS.2 generates a single centred *object*, so it cannot produce
# a kitchen. MoGe can, from a site photo.
RUN comfy model download \
      --url https://huggingface.co/Comfy-Org/MoGe/resolve/main/geometry_estimation/moge_2_vitl_normal_fp16.safetensors \
      --relative-path models/geometry_estimation \
      --filename moge_2_vitl_normal_fp16.safetensors

# BiRefNet cuts the subject out. Image-to-3D models reconstruct whatever is not
# background, so a product shot on a showroom floor becomes a product welded to a slab
# of floor unless this runs first.
RUN comfy model download \
      --url https://huggingface.co/Comfy-Org/BiRefNet/resolve/main/background_removal/birefnet.safetensors \
      --relative-path models/background_removal \
      --filename birefnet.safetensors

COPY workflows/ /workflows/
COPY build_api_workflows.py /build_api_workflows.py
COPY handler.py /handler.py

# Convert the vendor's UI-format templates to API format using THIS ComfyUI's own node
# definitions, and fail the build if anything does not line up. See the script's
# docstring for why a static conversion is not possible. This is the step that catches
# a renamed input or a node missing from this ComfyUI version -- on the builder, where
# the traceback is readable, rather than on a GPU billing by the second.
RUN cd /workflows && cp /build_api_workflows.py . \
 && python build_api_workflows.py \
 && rm build_api_workflows.py

# Same class of check as the sibling comfy worker: boot the whole graph engine on CPU
# and exit. A missing model file or a broken core node surfaces here.
RUN cd /comfyui && timeout 900 python main.py --quick-test-for-ci --cpu \
 && echo "comfyui boots clean"

RUN python -c "import ast; ast.parse(open('/handler.py').read()); print('handler parses OK')" \
 && python -c "import json,glob,sys; \
fs=glob.glob('/workflows/*.api.json'); \
sys.exit('no API workflows were built') if not fs else None; \
[json.load(open(f)) for f in fs]; \
print('api workflows valid:', fs)"

ENV WORKFLOW_DIR=/workflows \
    PYTHONUNBUFFERED=1

CMD ["/start.sh"]
