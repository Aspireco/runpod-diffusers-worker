# ComfyUI-based worker covering the production-polish lanes: cutouts, upscaling, frame
# interpolation and relighting.
#
# WHY THIS EXISTS. The hand-built workers in the sibling directories resolve their own
# dependencies from a bare PyTorch base, and that is where a full day went: torch silently
# upgraded to CUDA 13 wheels by a transitive dep, transformers API drift breaking an import,
# numpy refusing to build on Python 3.12. ComfyUI is a *pre-solved* environment — someone
# already fought those fights and pinned a set that coexists. Extending it is the job;
# re-deriving it was not.
#
# Kept hand-built: Z-Image, FLUX.2-klein, Qwen-Image, Wan 2.2. Those work, their images are
# small, and they cold-start in seconds. This is for the long tail.
#
# LICENCE NOTE: worker-comfyui is AGPL-3.0 and ComfyUI core is GPL-3.0. Running this
# internally triggers nothing. If it is ever exposed as a service clients log into, AGPL
# section 13 obliges offering them the source of the modified version. Recorded in
# DECISIONS.md.
FROM runpod/worker-comfyui:5.10.0-base

# HuggingFace pulls below are AUTHENTICATED when the CI secret is present.
# Anonymous pulls from pooled GitHub runner IPs hit HTTP 429 once several weight-baking
# lanes build concurrently. The token arrives as a BuildKit secret, not a build-arg, so
# it never lands in the image layer history. `|| true` keeps a local build working with
# no secret -- it just falls back to anonymous.
# Custom nodes, by Comfy Registry name. Deliberately NOT installed, all non-commercial:
#   ComfyUI-SUPIR          licence forbids "image analysis, enhancement, or modification
#                          for clients" -- a description of this business
#   ComfyUI-GIMM-VFI       S-Lab 1.0, non-commercial
#   ComfyUI-Rife-Tensorrt  CC-BY-NC-SA even though RIFE itself is MIT
RUN comfy-node-install \
      comfyui-frame-interpolation \
      comfyui-ic-light \
      comfyui-rmbg

# The frame-interpolation pack ships three non-commercial models in the same dropdown as the
# MIT ones (RIFE, FILM). An operator picking from a list cannot see a licence, so remove
# them from the image rather than relying on anyone reading this comment.
RUN rm -rf /comfyui/custom_nodes/*frame*interpolation*/vfi_models/m2m \
           /comfyui/custom_nodes/*frame*interpolation*/vfi_models/sepconv \
           /comfyui/custom_nodes/*frame*interpolation*/vfi_models/amt \
    && echo "removed non-commercial VFI models"

# Real-ESRGAN x4plus, BSD-3-Clause, from xinntao's own release. NOT 4x-UltraSharp or
# 4x-Remacri -- the two most-used upscalers in ComfyUI and both CC-BY-NC-SA.
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" \
    HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
    comfy model download \
      --url https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
      --relative-path models/upscale_models \
      --filename RealESRGAN_x4plus.pth

# IC-Light v1, background-conditioned. This is the checkpoint that relights a cut-out
# product to match a supplied scene -- the thing that makes a cedar bench look photographed
# in the room rather than pasted onto it. v2 is FLUX-based and non-commercial.
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" \
    HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
    comfy model download \
      --url https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fbc.safetensors \
      --relative-path models/unet \
      --filename iclight_sd15_fbc.safetensors

# SD1.5 base -- IC-Light's UNet offsets were computed against exactly this checkpoint, so
# anything else is a compatibility gamble. runwayml/stable-diffusion-v1-5 was deleted from
# HuggingFace; stable-diffusion-v1-5/stable-diffusion-v1-5 is the community relocation and
# is the canonical home now. Verified live before pinning: HTTP 200, 4.3GB, ungated.
#
# A previous revision pointed at stablediffusionapi/realistic-vision-v51 and 404'd the whole
# build -- that path does not exist. If a photographic finetune is ever wanted here,
# SG161222/Realistic_Vision_V5.1_noVAE is real (200, 2.1GB, same licence) but ships NO VAE,
# so it needs a separate VAE file; this one is self-contained.
#
# LICENCE: creativeml-openrail-m. Commercial use is permitted -- OpenRAIL is a use-based
# licence (it forbids specific harmful applications), not a non-commercial one. Distinct
# from the CC-BY-NC weights excluded above.
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" \
    HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
    comfy model download \
      --url https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors \
      --relative-path models/checkpoints \
      --filename v1-5-pruned-emaonly.safetensors

# Fail the BUILD rather than a billing GPU. ComfyUI's own CI flag boots the graph on CPU and
# exits -- a broken node install or a missing model surfaces here, where the log is readable,
# instead of as a crash-loop with no container output.
RUN cd /comfyui && timeout 600 python main.py --quick-test-for-ci --cpu \
    && echo "comfyui boots clean"
