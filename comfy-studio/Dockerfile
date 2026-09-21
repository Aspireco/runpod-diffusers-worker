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

# ---------------------------------------------------------------------------
# ComfyUI-LTXVideo, pinned. Lightricks' own pack: the Dub-It IC-LoRA (relip and
# re-voice a clip to new text, keeping the speaker), the x2/x4 pixel spatial
# upscalers, and the A2V graphs. Core already has the basic LTX nodes, so this
# is purely the extras.
#
# PINNED TO A COMMIT, AND CLONED FROM UPSTREAM, ON PURPOSE.
#
# `comfy-node-install ComfyUI-LTXVideo` would have been shorter and is wrong
# twice over. The registry entry for this pack carries an anonymous admin
# publisher id, latest_version null and 0 downloads (checked 2026-09-21) --
# everyone installs it by git URL, so the registry copy is not the real
# distribution. And the registry generally is not trustworthy for this: it
# hosts third-party FORKS of popular packs, frozen in time, outranking their
# upstreams on installs -- Mattabyte/ComfyUI-LTXVideo-Registry_Mattabyte is a
# fork stuck at 2025-03-12 with 44k installs, and comfyui-ltxvideolora has 75k
# installs behind a repository URL that 404s. A name search can quietly hand a
# render worker eighteen-month-old code.
#
# The pin also makes the image reproducible: without it, two builds of the same
# Dockerfile weeks apart are two different workers.
#
# LICENCE: LTX-2 Community License (5 Jan 2026), not Apache. Commercial use is
# granted below US$10M annual revenue; above that it needs a paid licence.
ARG LTXV_REF=dfb2786749af36f200ea023388dc729a3e106b42
RUN git clone https://github.com/Lightricks/ComfyUI-LTXVideo.git /comfyui/custom_nodes/ComfyUI-LTXVideo \
 && cd /comfyui/custom_nodes/ComfyUI-LTXVideo \
 && git checkout -q "${LTXV_REF}" \
 && pip install --no-cache-dir -r requirements.txt \
 && echo "ComfyUI-LTXVideo pinned at ${LTXV_REF}"
