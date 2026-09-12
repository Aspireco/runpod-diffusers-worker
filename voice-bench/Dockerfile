# Voice BENCHMARK worker: TTS-Audio-Suite, 19 engines behind one node vocabulary.
#
# WHY A SEPARATE IMAGE. The acceptance criterion is "the same script across every model".
# Doing that with one image per engine means nine images, nine cold starts, nine slightly
# different node vocabularies and nine chances to accidentally benchmark a configuration
# difference instead of a model. diodiogod/TTS-Audio-Suite (MIT, v5.8.10, 155k registry
# downloads, pushed 2026-09-05) wraps Qwen3-TTS, VibeVoice, CosyVoice3, Chatterbox,
# IndexTTS-2, Higgs Audio 2/3, F5-TTS, Echo-TTS, MOSS-TTS, Step Audio EditX, Dots TTS,
# OmniVoice and RVC behind one set of nodes. Same graph, swap the engine, compare.
#
# It also solves the RVC problem. Voice CONVERSION -- run a real performance through a
# model rather than synthesising from text -- has no installable ComfyUI node of its own:
# the registry's "ComfyUI-RVC" entry is an unclaimed stub with ZERO published versions, so
# `comfy-node-install ComfyUI-RVC` resolves and then installs nothing. The suite is the
# only maintained route to RVC inside ComfyUI, which is why this is not merely a test rig.
#
# THIS IS NOT A PRODUCTION LANE. Deliberately:
#   * No baked weights. Nine engines' checkpoints is >60GB and CI has ~45GB of runner disk.
#     Each engine pulls on first use -- fine for a benchmark, wrong for a cold-start budget.
#     Point HF_HOME at a network volume so the shootout pays each download once.
#   * Several engines it exposes are NOT usable commercially and must never be selected for
#     client work: F5-TTS weights are cc-by-nc-4.0, Higgs Audio v3 is research/non-
#     commercial, OpenAudio/Fish S2 is research-only, IndexTTS carries a bespoke bilibili
#     licence. The suite offers them in the same dropdown as the clean ones, exactly the
#     hazard ../comfy documents for the non-commercial VFI models. The difference is that
#     there they were deleted from the image; here they are the point of the image -- we
#     benchmark them to know what we are giving up. The licence column in
#     stack-docs/VOICE-BENCHMARK.md is the control, and the reason that column exists.
#   * Park it at --workers-max 0 the moment the shootout is done. Idle workers bill.
FROM runpod/worker-comfyui:5.10.0-base

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg sox espeak-ng git \
    && rm -rf /var/lib/apt/lists/*

# Pin the base image's torch BEFORE the suite installs anything. The suite declares 19
# engines' worth of dependencies and any one of them can satisfy a torch requirement with a
# PyPI CUDA 13 wheel -- those need driver >= 580, Runpod hosts run 570/575, and CUDA init
# then dies before a single log line. That reads as a crash-loop with no cause and has
# already cost this project a day. Same guard as ../music and ../tts.
RUN python -c "import torch, torchvision, torchaudio; \
open('/tmp/constraints.txt','w').write(''.join(f'{n}=={v}\n' for n,v in \
[('torch',torch.__version__),('torchvision',torchvision.__version__),('torchaudio',torchaudio.__version__)]))" \
    && cat /tmp/constraints.txt \
    && cp /tmp/constraints.txt /constraints.txt

# The suite targets a transformers 5.x environment while the base ships 4.x, and it expects
# to resolve that itself -- so transformers is deliberately NOT pinned here, matching the
# reasoning in ../tts (pin torch, which breaks CUDA; let the package choose the rest).
RUN PIP_CONSTRAINT=/constraints.txt comfy-node-install tts_audio_suite

# Assert torch survived. This runs immediately after the only step that could have replaced
# it, so a failure here names the culprit instead of surfacing as a silent GPU crash-loop.
RUN python -c "import torch; v=torch.__version__; print('torch', v); \
assert 'cu13' not in v, f'torch was replaced by a suite dependency: {v}'"

# Teach the handler to return audio at all -- worker-comfyui 5.10.0 walks only the "images"
# output key and SaveAudio* nodes emit under "audio". Without this the entire benchmark
# would report nine successful runs and produce zero audio files.
COPY patch_audio_outputs.py /patch_audio_outputs.py
RUN python /patch_audio_outputs.py \
    && python -c "import ast; ast.parse(open('/handler.py').read()); print('handler parses OK')" \
    && grep -q 'audio passthrough' /handler.py

COPY workflows /workflows

# Boot the graph on CPU and exit. With 19 engines' imports this is the step most likely to
# catch a dependency conflict, and it catches it here -- where the log is readable -- rather
# than on a billing GPU with no container output. Longer timeout than the sibling images
# because the suite's import surface is large.
RUN cd /comfyui && timeout 1800 python main.py --quick-test-for-ci --cpu \
    && echo "comfyui boots clean"

# Prove the suite actually registered, and that RVC specifically came with it -- RVC is the
# reason this image exists beyond benchmarking, and a partial install that drops it would
# otherwise be invisible until a conversion job failed.
RUN cd /comfyui && timeout 1800 python /dev/stdin <<'PY'
import sys
sys.path.insert(0, "/comfyui")
print("[boot] importing nodes", flush=True)
import nodes
nodes.init_extra_nodes(init_custom_nodes=True)
have = sorted(nodes.NODE_CLASS_MAPPINGS)
suite = [n for n in have if "tts" in n.lower() or "voice" in n.lower() or "rvc" in n.lower()]
assert len(suite) >= 5, f"tts_audio_suite barely registered. got: {suite}"
assert any("rvc" in n.lower() for n in have), (
    "RVC nodes absent -- the suite is the only maintained ComfyUI route to voice "
    f"conversion, so this image is pointless without them. registered: {suite}")
print(f"[boot] tts_audio_suite OK, {len(suite)} voice nodes", flush=True)
print("[boot] sample:", suite[:25], flush=True)
PY

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1
