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
# Downloads go through fetch_model.sh rather than `comfy model download`: that command
# retries three times and gives up, and this image pulls ~9.6GB from HuggingFace, which
# after a few rebuilds in one hour earns a 429 Too Many Requests. A build already died
# that way on the MoGe weights after TRELLIS.2 had succeeded. The helper backs off long
# enough to outlast a rate-limit, resumes partial transfers, and asserts each file's
# size and magic bytes -- because the dangerous failure is not an error, it is an HTML
# error page written to a .safetensors path with exit code 0, which would fail later on
# a billing GPU instead of here.
COPY fetch_model.sh /usr/local/bin/fetch-model
RUN chmod +x /usr/local/bin/fetch-model \
 && (command -v curl >/dev/null || (apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*))

# HF_TOKEN arrives as a BuildKit SECRET, never a build-arg: a build-arg is recorded in
# the image layer history, so anyone pulling the image could read the token back out.
# `|| true` keeps local builds working with no secret -- these repos are all public, so
# the token only buys a higher rate limit, which is precisely what the 429 above was.
#
# Separate layers on purpose: with the GHA cache, a 429 on a later model does not
# re-pull the 5.25GB that already landed.
#
# int8_convrot over bf16: 5.25GB vs 10.34GB for output the benchmark could not tell
# apart on product shots, and it is the difference between a cold start that pulls
# 9GB and one that pulls 14GB. bf16 is a one-line change if a client ever needs it.
RUN --mount=type=secret,id=hf_token \
    export HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)"; \
    fetch-model \
      https://huggingface.co/Comfy-Org/TRELLIS.2/resolve/main/diffusion_models/trellis_2_int8_convrot.safetensors \
      /comfyui/models/diffusion_models trellis_2_int8_convrot.safetensors 5000000000

RUN --mount=type=secret,id=hf_token \
    export HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)"; \
    fetch-model \
      https://huggingface.co/Comfy-Org/TRELLIS.2/resolve/main/vae/trellis_2_shape_vae_bf16.safetensors \
      /comfyui/models/vae trellis_2_shape_vae_bf16.safetensors 1000000000  && fetch-model \
      https://huggingface.co/Comfy-Org/TRELLIS.2/resolve/main/vae/trellis_2_texture_vae_bf16.safetensors \
      /comfyui/models/vae trellis_2_texture_vae_bf16.safetensors 900000000

# The image encoder. Filename matters: the vendor template's CLIPVisionLoader asks for
# dino_v3_L_naf_fp32.safetensors, which lives in Comfy-Org/Pixal3D, NOT in
# Comfy-Org/TRELLIS.2 (that repo ships a differently-named dino_v3_vit_l.safetensors).
# Pulling the TRELLIS.2 one and renaming it produces a graph that loads and generates
# noise, because they are not the same tensor layout.
RUN --mount=type=secret,id=hf_token \
    export HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)"; \
    fetch-model \
      https://huggingface.co/Comfy-Org/Pixal3D/resolve/main/clip_vision/dino_v3_L_naf_fp32.safetensors \
      /comfyui/models/clip_vision dino_v3_L_naf_fp32.safetensors 1100000000

# MoGe-2 estimates per-pixel geometry and camera FOV from one photo. In the TRELLIS.2
# graph it supplies the FOV that sets object proportions. On its own it is the
# room-scale lane: TRELLIS.2 generates a single centred *object*, so it cannot produce
# a kitchen. MoGe can, from a site photo.
RUN --mount=type=secret,id=hf_token \
    export HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)"; \
    fetch-model \
      https://huggingface.co/Comfy-Org/MoGe/resolve/main/geometry_estimation/moge_2_vitl_normal_fp16.safetensors \
      /comfyui/models/geometry_estimation moge_2_vitl_normal_fp16.safetensors 600000000

# BiRefNet cuts the subject out. Image-to-3D models reconstruct whatever is not
# background, so a product shot on a showroom floor becomes a product welded to a slab
# of floor unless this runs first.
RUN --mount=type=secret,id=hf_token \
    export HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)"; \
    fetch-model \
      https://huggingface.co/Comfy-Org/BiRefNet/resolve/main/background_removal/birefnet.safetensors \
      /comfyui/models/background_removal birefnet.safetensors 400000000

COPY workflows/ /workflows/
COPY build_api_workflows.py /build_api_workflows.py
COPY handler.py /handler.py

# Convert the vendor's UI-format templates to API format using THIS ComfyUI's own node
# definitions, and fail the build if anything does not line up. See the script's
# docstring for why a static conversion is not possible. This is the step that catches
# a renamed input or a node missing from this ComfyUI version -- on the builder, where
# the traceback is readable, rather than on a GPU billing by the second.
RUN WORKFLOW_DIR=/workflows python /build_api_workflows.py

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
