# Voice worker: Qwen3-TTS. The bilingual EN/FR ad-read lane, and the default for new work.
#
# WHY THIS IS THE PRIMARY LANE, not VibeVoice. The brief seeded VibeVoice as the headline
# model and Qwen3-TTS as a thing to test. Verification reversed that:
#
#   * Qwen3-TTS weights are genuinely open. That was the live question -- every previous
#     Qwen TTS release was DashScope-API-only, and Qwen3-TTS-Flash still is. But
#     Qwen/Qwen3-TTS-12Hz-* are real downloadable safetensors, apache-2.0, ungated,
#     3.7M downloads/month on the 1.7B-Base alone. Verified against the Qwen HF org
#     listing and QwenLM/Qwen3-TTS, which carries an actual Apache LICENSE file.
#   * French is a first-class language, not a checkbox. Ten languages, French among them,
#     with a published French WER of 2.858 -- the best number found for any open model.
#     The business is bilingual EN/FR Canadian, so this is the axis that matters.
#   * No watermark, no disclaimer, anywhere in the code path.
#   * VibeVoice, by contrast, carries a licence conflict serious enough to keep it off
#     client work by default -- MIT tag, NO LICENSE file in the weight repo, and a README
#     that says "limited to research purpose use" and "we do not recommend using VibeVoice
#     in commercial or real-world applications". See DECISIONS.md Part 1. It stays in
#     ../voice-comfy for the long-form lane it genuinely wins, flagged.
#
# Models here, all apache-2.0, all ungated, verified against the HF API on 2026-09-12:
#   Qwen3-TTS-Tokenizer-12Hz        shared audio tokenizer, required by both below
#   Qwen3-TTS-12Hz-1.7B-Base        3-second zero-shot voice cloning -- the brand-voice lane
#   Qwen3-TTS-12Hz-1.7B-CustomVoice 9 preset timbres + natural-language delivery direction
# VoiceDesign (invent a voice from a text description) is a deliberate omission: another
# ~4.5GB for a capability we can reach by cloning a reference instead. Opt in with
# QWEN_TTS_EXTRA if a brand ever needs a voice that does not exist yet.
FROM runpod/worker-comfyui:5.10.0-base

# Registry id is CASE-SENSITIVE and counter-intuitive here. Three plausible spellings all
# resolve to real but DIFFERENT node packs:
#   qwen3-tts-comfyui   flybirdxx   54k downloads, v1.0.7, pushed 2026-09-06  <- this one
#   comfyui-qwen3-tts   DarioFT     19k downloads, v1.7.0
#   ComfyUI-QwenTTS     1038lab     28k downloads, v1.1.4
# flybirdxx wins on adoption (1.9k GitHub stars) and recency, and is the only one of the
# three exposing cloning, CustomVoice, VoiceDesign, a dialogue node and a fine-tune node
# from one pack.
RUN comfy-node-install qwen3-tts-comfyui

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg sox \
    && rm -rf /var/lib/apt/lists/*

# comfy-node-install does NOT reliably install a node pack's own requirements -- the
# sibling ../voice-comfy build proved it by dying on a missing kokoro_onnx. This node
# imports librosa, soundfile, tiktoken and sentencepiece at module scope, so without them
# it fails to register at RUNTIME and the only symptom is a workflow referencing a node
# class that does not exist, discovered on a billing GPU.
#
# Pin torch first: the node declares transformers>=4.57.0 and a resolver satisfying a torch
# requirement with a PyPI CUDA 13 wheel needs driver >= 580 while Runpod hosts run 570/575,
# which kills CUDA before a single log line. --prefer-binary rather than --only-binary
# because not every one of these ships a wheel at every version, and a hard refusal here
# fails the build for a package that would have compiled fine.
#
# NOTE `sox` here is the PYTHON package, which is NOT the apt `sox` binary installed
# above -- the first build installed the binary and the node still died on
# `ModuleNotFoundError: No module named 'sox'`. Both are needed: pysox shells out to
# the binary, so neither alone is sufficient.
#
# onnxruntime: the node imports it UNCONDITIONALLY, and leaving it out cost a build --
# both node classes failed to register with `ModuleNotFoundError: No module named
# 'onnxruntime'`. The plain CPU wheel, NOT onnxruntime-gpu, which drags in a second CUDA
# runtime that can collide with torch's for a component doing no heavy lifting here.
#
# onnxruntime-openvino from the node's requirements is still skipped: it is an Intel
# CPU inference runtime, useless on an NVIDIA serverless host, and it drags in a second
# runtime stack for nothing.
RUN python -c "import torch, torchvision, torchaudio; \
open('/tmp/constraints.txt','w').write(''.join(f'{n}=={v}\n' for n,v in \
[('torch',torch.__version__),('torchvision',torchvision.__version__),('torchaudio',torchaudio.__version__)]))" \
    && cat /tmp/constraints.txt \
    && pip install --no-cache-dir -c /tmp/constraints.txt --prefer-binary \
         "transformers>=4.57.0,<5" "librosa" "soundfile" "einops" "tiktoken" \
         "sentencepiece" "accelerate" "scipy>=1.11.0" "safetensors" "sox" "onnxruntime" \
    && python -c "import torch; v=torch.__version__; print('torch', v); \
assert 'cu13' not in v, f'torch was replaced: {v}'" \
    && python -c "import librosa, soundfile, tiktoken, sentencepiece, transformers; \
print('qwen3-tts node deps OK, transformers', transformers.__version__)" \
    && python -c "import sox, onnxruntime; print('python sox + onnxruntime OK')"

# Use the node's OWN downloader rather than a hand-rolled file list. It resolves
# ComfyUI/models/qwen-tts/<repo-basename>/ via snapshot_download, which is the exact
# layout its loader globs for. Hand-placing files here means silently re-deriving a layout
# that the node author is free to change in the next release. Passing --target explicitly
# because the script's auto-detection walks up from __file__ and the registry install path
# is not guaranteed to be two levels under /comfyui.
#
# HF_HUB_ENABLE_HF_TRANSFER is deliberately NOT set: it needs a rust extension that is not
# in this base image, and when it is missing huggingface_hub raises rather than falling
# back -- a build failure that reads as a network problem.
# AUTHENTICATED when the CI secret is present. Anonymous pulls from pooled GitHub
# runner IPs hit HTTP 429 once several weight-baking lanes build concurrently -- this
# lane hit it three times. The token arrives as a BuildKit secret, not a build-arg, so
# it never lands in the image layer history; `|| true` keeps a local build working with
# no secret, falling back to anonymous. The backoff below stays regardless: a higher
# rate limit is not an absent one.
#
# Retried with backoff for the same reason as ../voice-comfy: HuggingFace 429s these
# pulls from GitHub runner IPs, several lanes in this repo build concurrently and share
# them, and snapshot_download surfaces the 429 as a hard failure. It succeeded on an
# earlier build and failed the neighbouring lane in the same minute, which is exactly
# the signature of a shared limiter rather than a broken URL. snapshot_download resumes
# rather than restarting, so a retry picks up where the last attempt stopped.
RUN --mount=type=secret,id=hf_token \
    export HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" ; \
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" ; \
    NODE_DIR="$(find /comfyui/custom_nodes -maxdepth 1 -iname '*qwen*tts*' | head -1)" \
    && test -n "$NODE_DIR" || (echo "qwen3-tts node not installed" && exit 1) \
    && ok=0 \
    && for delay in 0 30 60 120 240 480 ; do \
         [ $delay -gt 0 ] && { echo "retrying model download in ${delay}s" ; sleep $delay ; } ; \
         if python "$NODE_DIR/download_models.py" --target /comfyui/models/qwen-tts ; then \
           ok=1 ; break ; \
         fi ; \
       done \
    && { [ $ok -eq 1 ] || { echo "FAILED after backoff" ; exit 1 ; } ; }

# The downloader pulls the 0.6B pair only with a flag, and VoiceDesign is in its defaults.
# Drop VoiceDesign to keep the image near 10GB rather than 15GB. If the default set ever
# stops including it this is a no-op, which is why it is `rm -rf` and not a hard failure.
RUN rm -rf /comfyui/models/qwen-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
    && du -sh /comfyui/models/qwen-tts/* || true

# Teach the handler to return audio at all. worker-comfyui 5.10.0 walks only the "images"
# output key; SaveAudio* nodes emit under "audio", so without this every TTS job returns
# SUCCESS with an empty payload. Shared with ../voice-comfy -- same bug, same fix.
COPY patch_audio_outputs.py /patch_audio_outputs.py
RUN python /patch_audio_outputs.py \
    && python -c "import ast; ast.parse(open('/handler.py').read()); print('handler parses OK')" \
    && grep -q 'audio passthrough' /handler.py

COPY workflows /workflows

# ---------------------------------------------------------------------------------------
# Build-time checks that fail the BUILD rather than a billing GPU.
RUN python /dev/stdin <<'PY'
import glob, json, os
root = "/comfyui/models/qwen-tts"
need = ["Qwen3-TTS-Tokenizer-12Hz",
        "Qwen3-TTS-12Hz-1.7B-Base",
        "Qwen3-TTS-12Hz-1.7B-CustomVoice"]
for n in need:
    d = os.path.join(root, n)
    assert os.path.isdir(d), f"missing model dir {d} -- downloader layout changed"
    size = sum(os.path.getsize(p) for p in glob.glob(os.path.join(d, "**", "*"), recursive=True)
               if os.path.isfile(p))
    # A 302 to an HTML error page lands as a plausible-looking small file. Anything under
    # 100MB for these three is a failed download wearing a model's name.
    assert size > 1e8, f"{n} is only {size/1e6:.0f}MB -- download returned an error page"
    cfg = os.path.join(d, "config.json")
    if os.path.exists(cfg):
        json.load(open(cfg))                  # HTML masquerading as JSON dies here
    print(f"[check] {n}: {size/1e9:.2f}GB")
PY

# Boot the whole graph on CPU and exit. A broken node install or a bad pin surfaces here,
# where the log is readable, instead of as a crash-loop with no container output.
RUN cd /comfyui && timeout 900 python main.py --quick-test-for-ci --cpu \
    && echo "comfyui boots clean"

# comfy-node-install can succeed while the node fails to import, and the only symptom is
# the workflow JSON referencing a node class that does not exist -- at request time, on a
# billing GPU. Assert the exact class names the shipped workflows reference.
RUN cd /comfyui && timeout 900 python /dev/stdin <<'PY'
import sys
sys.path.insert(0, "/comfyui")
# ComfyUI parses sys.argv at import time via comfy.cli_args, and importing `nodes`
# pulls in model_management, which calls torch.cuda.current_device() at module level.
# CI runners have no GPU, so without --cpu this dies on "Found no NVIDIA driver"
# before a single node is registered -- a build failure that says nothing about the
# nodes we are actually checking. The main.py check above already passes --cpu on its
# command line; this is the same flag, delivered the only way an inline script can.
sys.argv = ["main.py", "--cpu"]
# ...and setting sys.argv alone is NOT enough, which the previous build proved by
# failing here anyway. comfy/cli_args.py ends with `if comfy.options.args_parsing:
# args = parser.parse_args()` else `parser.parse_args([])`. That flag defaults to
# False, so an inline script's --cpu is silently discarded and args.cpu stays False.
# main.py works only because its first two lines flip the flag. These are those two
# lines, and they must come before `import nodes` pulls in model_management.
import comfy.options
comfy.options.enable_args_parsing()
print("[boot] importing nodes", flush=True)
import nodes
# init_extra_nodes is `async def` in current ComfyUI (main.py does
# asyncio_loop.run_until_complete on it). Calling it bare returns a coroutine and
# registers NOTHING -- silently. That is how this check reported 'nodes missing'
# while the step above it logged 'VibeVoice nodes registered successfully': the node
# was fine, the check was asserting against an empty registry it never populated.
# inspect.isawaitable keeps this working if a future base pins a sync version.
import asyncio, inspect
_res = nodes.init_extra_nodes(init_custom_nodes=True)
if inspect.isawaitable(_res):
    asyncio.run(_res)
have = set(nodes.NODE_CLASS_MAPPINGS)
want = {"FB_Qwen3TTSCustomVoice", "FB_Qwen3TTSVoiceClone"}
missing = want - have
assert not missing, (f"Qwen3-TTS nodes missing: {missing}\n"
                     f"qwen-ish registered: {sorted(n for n in have if 'qwen' in n.lower())}")
assert any("Audio" in n for n in have), "no SaveAudio-class node registered"
print("[boot] Qwen3-TTS OK:", sorted(n for n in have if n.startswith("FB_Qwen3TTS")), flush=True)
PY

# Opt-in extras, pulled at runtime to a network volume rather than baked. VoiceDesign is
# the interesting one; the 0.6B pair is the answer if a cheaper GPU tier is ever needed.
ENV QWEN_TTS_EXTRA="" \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1
