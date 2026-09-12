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

# build-essential is here and NOT in the sibling images on purpose. Across 19 engines'
# dependencies at least one has no wheel for this Python, and the base image ships no
# compiler -- the first build died on `No such file or directory: 'x86_64-linux-gnu-g++'`
# inside a wheel build. The production lanes stay compiler-free (a compiler in a
# serverless image is image size and attack surface for nothing); this rig earns it.
#
# python3-dev is the second half, and omitting it cost another build: with the compiler
# present but no headers, pyworld and bitpack both died on `fatal error: Python.h: No
# such file or directory`. A compiler alone cannot build a Python extension.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg sox espeak-ng git build-essential python3-dev \
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

# comfy-node-install does not reliably install a pack's own requirements -- the sibling
# ../voice-comfy build proved it by dying on a missing kokoro_onnx. Install this pack's
# requirements.txt explicitly if one shipped. --prefer-binary rather than --only-binary:
# across 19 engines' dependencies something will lack a wheel at some version (a sibling
# worker hit exactly that with docopt), and a hard refusal fails the build for a package
# that would have compiled fine. Still constrained, so torch cannot be replaced.
#
# `|| true` on the install ONLY: with 19 engines, one optional extra failing to resolve
# should not sink the image, and the boot check below is the real gate -- it fails the
# build if the suite did not actually register. That is the difference between tolerating
# a missing optional dep and shipping a broken image.
RUN NODE_DIR="$(find /comfyui/custom_nodes -maxdepth 1 -iname '*tts*audio*suite*' -o -maxdepth 1 -iname '*TTS-Audio-Suite*' | head -1)" \
    && echo "suite at: ${NODE_DIR:-<not found>}" \
    && if [ -f "$NODE_DIR/requirements.txt" ]; then \
         pip install --no-cache-dir -c /constraints.txt --prefer-binary \
           -r "$NODE_DIR/requirements.txt" || true ; \
       else echo "no requirements.txt shipped with the pack" ; fi

# The pack's own requirements.txt did not install: comfy-node-install's cm_cli choked on
# it with `Failed to parse: numpy>=1.26.4,<2.3.0  # Compatible with both numpy ...` --
# an inline comment after a version spec that its parser cannot handle. So nothing from
# that file landed, and the boot check then failed on missing librosa and soundfile.
# pip's own parser handles the file fine, which is why the -r install above is the
# fallback; these are the module-scope imports named in the failure, pinned belt-and-
# braces so a silently-skipped requirements file cannot take the image down again.
RUN pip install --no-cache-dir -c /constraints.txt --prefer-binary \
      librosa soundfile audio-separator \
    && python -c "import librosa, soundfile; print('bench audio deps OK')"

# Assert torch survived. Placed after BOTH install steps above -- either could have replaced
# it -- so a failure here names the culprit instead of surfacing as a silent GPU crash-loop.
# This is also why the `|| true` above is survivable: a dependency that quietly swapped
# torch on its way out still fails the build, here.
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
have = sorted(nodes.NODE_CLASS_MAPPINGS)
suite = [n for n in have if "tts" in n.lower() or "voice" in n.lower() or "rvc" in n.lower()]
assert len(suite) >= 5, f"tts_audio_suite barely registered. got: {suite}"
assert any("rvc" in n.lower() for n in have), (
    "RVC nodes absent -- the suite is the only maintained ComfyUI route to voice "
    f"conversion, so this image is pointless without them. registered: {suite}")
print(f"[boot] tts_audio_suite OK, {len(suite)} voice nodes", flush=True)
print("[boot] sample:", suite[:25], flush=True)
# Validate the SHIPPED WORKFLOWS against the registry we just built.
#
# This is the check that matters, and it was missing. The assertions above prove that
# *some* expected nodes exist; they say nothing about whether the JSON files this image
# actually ships reference real class names. They did not: workflows/kokoro-narration.json
# invented a "KokoroSpeech" node that has never existed (the pack registers KokoroSpeaker
# and KokoroGenerator), and every earlier build passed anyway. That error would have
# surfaced as a 400 on the first real request, on a billing GPU.
#
# Node packs rename classes between versions, so this also turns a silent upstream rename
# into a build failure with the old and new names side by side.
import json, glob, os
wf_errors = []
for path in sorted(glob.glob("/workflows/*.json")):
    graph = json.load(open(path))
    for node_id, node in graph.items():
        if node_id.startswith("_") or not isinstance(node, dict):
            continue                      # _comment / _meta documentation keys
        ct = node.get("class_type")
        if not ct:
            wf_errors.append(f"{os.path.basename(path)} node {node_id}: no class_type")
        elif ct not in have:
            near = sorted(n for n in have if ct.lower()[:6] in n.lower())[:5]
            wf_errors.append(
                f"{os.path.basename(path)} node {node_id}: class_type {ct!r} is not "
                f"registered in this image. closest: {near or 'nothing similar'}")
assert not wf_errors, "shipped workflows reference nodes this image does not have:\n  " + \
    "\n  ".join(wf_errors)
print(f"[boot] all class_types in {len(glob.glob('/workflows/*.json'))} shipped workflow(s) exist", flush=True)
PY

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1
