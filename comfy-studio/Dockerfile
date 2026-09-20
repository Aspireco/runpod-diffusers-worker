# Studio video worker: LTX-2.5 and MiniMax H3 graphs, models from the network volume.
#
# WHY A SEPARATE IMAGE. runpod/worker-comfyui:5.10.0-base ships ComfyUI 0.34.0. MiniMax H3
# has been in core since 0.30.0, but LTX-2.5's Gemma4 text encoder needs the
# `ltxav_gemma4_tokenizer` that arrived in 0.35.0 -- on 0.34 the CLIPLoader dies with a bare
# "invalid tokenizer" (measured on the pod, 2026-09-18). So this image is the base with
# ComfyUI moved to v0.35.0 and nothing else: no nodes, no weights. The weights live on the
# studio network volume (mos-studio-models, CA-MTL-3) under /runpod-volume/models/..., which
# worker-comfyui reads automatically:
#   diffusion_models/  ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors
#                      minimax_h3_fl2va_pruned_fp8_scaled.safetensors
#   text_encoders/     gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors
#                      qwen3vl_32b_minimax_h3_int8_convrot.safetensors
#   vae/               ltx-2.5-video-vae-conv-bf16.safetensors  (the -conv- one; the plain
#                      bf16 file is a transformer decoder ComfyUI misdetects as an SD VAE)
#                      minimax_h3_video_vae_fp16.safetensors, minimax_h3_audio_vae_fp32.safetensors
#   checkpoints/       ltx-2.5-audio-vae-bf16.safetensors  (LTXVAudioVAELoader reads checkpoints/)
#   loras/             minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors
#
# LICENCE NOTE: same as ../comfy -- worker-comfyui is AGPL-3.0, ComfyUI core GPL-3.0; the
# models carry their own terms (LTX-2.x Community: free under US$10M revenue; MiniMax H3
# Community: outputs may not be shown in the EU/UK/KR/USA, credit required). The gateway's
# registry enforces the territory rule per request.
FROM runpod/worker-comfyui:5.10.0-base

ARG COMFYUI_REF=v0.35.0
# The base installs ComfyUI as a git checkout at /comfyui (comfy-cli). Move it to the tag
# and let its own requirements resolve the few new pins; torch stays as the base pinned it.
RUN cd /comfyui \
 && git fetch --tags --depth 1 origin "${COMFYUI_REF}" \
 && git checkout -q "${COMFYUI_REF}" \
 && pip install --no-cache-dir -r requirements.txt \
 && python -c "import comfyui_version; print('ComfyUI', comfyui_version.__version__)"
