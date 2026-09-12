# Wan 2.2 S2V-14B -- speech-to-video. Give it a voice track and a reference image and it
# returns a lip-synced performance.
#
# WHAT THIS IS NOT: S2V does not generate a voice. It CONSUMES one. The pipeline is
#   Qwen3-TTS (apache-2.0, the voice-qwen lane) -> audio -> S2V -> talking video
# which is why this lane pairs with the voice lanes rather than replacing them.
#
# WHY COMFYUI AND NOT DIFFUSERS: there is no WanSpeechToVideoPipeline in diffusers -- not in
# 0.37, 0.38, 0.39, 0.40, nor main; checked directly against each tag's __init__.py. The
# upstream Wan-AI/Wan2.2-S2V-14B repo also ships the ORIGINAL Wan layout (raw shards,
# Wan2.1_VAE.pth, models_t5_umt5-xxl-enc-bf16.pth) with no model_index.json, so
# DiffusionPipeline.from_pretrained cannot read it either. ComfyUI has native S2V nodes.
#
# LICENCE: apache-2.0 on every file below, from Comfy-Org's repack of Wan's own weights.
# Wan is the only top-tier video family with no territory clause, no revenue cap and no
# attribution requirement -- which is why it is the lane that can actually ship.
FROM runpod/worker-comfyui:5.10.0-base

# HuggingFace pulls are AUTHENTICATED via the CI BuildKit secret. Anonymous pulls from
# pooled GitHub runner IPs hit HTTP 429 once several weight-baking lanes build at once --
# voice-comfy failed three times on exactly that. `|| true` keeps a local build working
# with no secret; it just falls back to anonymous.
ARG REPO=https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files

# The transformer, bf16. 32.6GB. The fp8_scaled build is half the size and exists at
# split_files/diffusion_models/wan2.2_s2v_14B_fp8_scaled.safetensors if a 48GB card is ever
# the target -- bf16 is chosen here because this lane runs on an 80GB card.
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" \
    HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
    comfy model download \
      --url ${REPO}/diffusion_models/wan2.2_s2v_14B_bf16.safetensors \
      --relative-path models/diffusion_models \
      --filename wan2.2_s2v_14B_bf16.safetensors

# wav2vec2 is what makes this lane speech-driven: it encodes the input audio into the
# conditioning the transformer animates against. Without it S2V has nothing to sync to.
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" \
    HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
    comfy model download \
      --url ${REPO}/audio_encoders/wav2vec2_large_english_fp16.safetensors \
      --relative-path models/audio_encoders \
      --filename wav2vec2_large_english_fp16.safetensors

RUN --mount=type=secret,id=hf_token \
    HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" \
    HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
    comfy model download \
      --url ${REPO}/text_encoders/umt5_xxl_fp16.safetensors \
      --relative-path models/text_encoders \
      --filename umt5_xxl_fp16.safetensors

# S2V uses the Wan 2.1 VAE, not the 2.2 one. Both ship in this repo and the names are one
# character apart, so this is an easy and silent mistake to make.
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" \
    HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
    comfy model download \
      --url ${REPO}/vae/wan_2.1_vae.safetensors \
      --relative-path models/vae \
      --filename wan_2.1_vae.safetensors

# Fail the BUILD, not a billing GPU, if a download silently produced a stub. `comfy model
# download` has returned success on a truncated file before, so check real sizes.
RUN python - <<'PY'
import os, sys
want = {
    "/comfyui/models/diffusion_models/wan2.2_s2v_14B_bf16.safetensors": 30e9,
    "/comfyui/models/audio_encoders/wav2vec2_large_english_fp16.safetensors": 0.5e9,
    "/comfyui/models/text_encoders/umt5_xxl_fp16.safetensors": 10e9,
    "/comfyui/models/vae/wan_2.1_vae.safetensors": 0.2e9,
}
bad = []
for p, floor in want.items():
    n = os.path.getsize(p) if os.path.exists(p) else 0
    print(f"  {n/1e9:7.2f} GB  {p}")
    if n < floor:
        bad.append(f"{p} is {n/1e9:.2f} GB, expected at least {floor/1e9:.2f} GB")
if bad:
    sys.exit("TRUNCATED OR MISSING:\n  " + "\n  ".join(bad))
print("all four S2V files present and full size")
PY

# ComfyUI's node registry is the real test: the S2V nodes must exist or the graph cannot be
# built at runtime. init_extra_nodes is `async def` in current ComfyUI -- calling it bare
# returns a coroutine, registers NOTHING and raises nothing, so an assertion written that
# way silently tests an empty registry. Await it.
RUN cd /comfyui && python - <<'PY'
import asyncio, inspect, sys
# --cpu must be in sys.argv BEFORE enable_args_parsing(), or ComfyUI initialises CUDA and
# dies with "Found no NVIDIA driver" on a GPU-less CI runner. enable_args_parsing() alone is
# not enough -- it only tells comfy.cli_args that argv is worth reading; the flag still has
# to be there to read.
sys.argv = [sys.argv[0], "--cpu"]
import comfy.options
comfy.options.enable_args_parsing()
import nodes
r = nodes.init_extra_nodes()
if inspect.isawaitable(r):
    asyncio.run(r)
have = set(nodes.NODE_CLASS_MAPPINGS)
need = ["WanSoundImageToVideo", "AudioEncoderLoader", "AudioEncoderEncode"]
missing = [n for n in need if n not in have]
print("registered node types:", len(have))
for n in need:
    print(("  OK      " if n in have else "  MISSING ") + n)
if missing:
    raise SystemExit("S2V nodes absent from this ComfyUI build: " + ", ".join(missing))
print("S2V node set present")
PY
