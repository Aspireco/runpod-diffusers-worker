# Voice worker: VibeVoice (long-form multi-speaker) + Kokoro (bulk narration), on ComfyUI.
#
# WHY COMFYUI. Same reason as ../comfy: worker-comfyui is a pre-solved dependency set.
# The sibling hand-built ../tts worker failed five builds on source compiles before it was
# forced to wheels-only, and VibeVoice's dependency surface is worse -- diffusers +
# transformers + bitsandbytes + peft, all pinned against each other by the node author.
# Extending a solved environment is the job; re-solving it is not.
#
# WHAT IS DELIBERATELY ABSENT: comfyui-f5-tts. It is the most-installed TTS node in the
# registry (76k downloads) and the node's own code is MIT -- but the weights it pulls,
# SWivid/F5-TTS, are **cc-by-nc-4.0**, because they were trained on Emilia. Non-commercial
# weights behind a permissive node is exactly the trap ../comfy hit with SUPIR. Verified
# against the HF API, not the README. Recorded in DECISIONS.md.
#
# LICENCE: worker-comfyui is AGPL-3.0, ComfyUI core GPL-3.0 -- internal use triggers
# nothing, exposing it as a service clients log into triggers AGPL s13. Already recorded in
# DECISIONS.md under ../comfy. Everything added here is MIT or Apache-2.0.
FROM runpod/worker-comfyui:5.10.0-base

# ---------------------------------------------------------------------------------------
# Custom nodes, by exact Comfy Registry id. These ids are CASE-SENSITIVE: the registry has
# no entry for "comfyui-vibevoice", only "ComfyUI-VibeVoice". Two of the four plausible
# spellings 404, so the case is load-bearing, not cosmetic.
#
#   VibeVoice-ComfyUI  (publisher enemyx, MIT, v1.8.1, 105k downloads, active 2026-02)
#     Chosen over wildminder/ComfyUI-VibeVoice (64k downloads) which has had no commit
#     since 2025-09 and has no LoRA support. Enemyx has LoRA (peft), 4-bit NF4 and 8-bit
#     quantisation, and -- the reason it matters here -- automatic chunking at
#     max_words_per_chunk, which is the documented mitigation for VibeVoice rushing its
#     delivery on long passages.
#
#   comfyui-kokoro     (publisher stavsap, Apache-2.0, v1.1.4)
#     Kokoro-82M. 82M params, ONNX, runs on CPU, ~210x realtime. French included
#     (fr-fr). No cloning -- 54 fixed voices. This is the bulk-narration lane where
#     paying VibeVoice's diffusion cost per word would be absurd.
RUN comfy-node-install \
      VibeVoice-ComfyUI \
      comfyui-kokoro

# espeak-ng is Kokoro's grapheme-to-phoneme backend for the non-English voices, French
# included. Without it the English voices work and French silently produces English
# phonemes -- audible nonsense rather than an error, so install it rather than discover it.
# ffmpeg backs the mp3/opus encoders behind SaveAudioMP3 / SaveAudioOpus.
RUN apt-get update && apt-get install -y --no-install-recommends espeak-ng ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------------------
# VibeVoice-1.5B, MIT, from Microsoft's own repo -- which is live and ungated as of
# 2026-09-12, contrary to the widely repeated claim that it was pulled. (What was pulled and
# never restored is microsoft/VibeVoice-Large; aoi-ot/VibeVoice-Large, also MIT, is the
# surviving copy of that one.)
#
# 1.5B rather than the 7B despite the 7B having better prosody: the 1.5B has the LONGER
# context (64K vs 32K) and the longer audio ceiling (~90 min vs ~45), it fits in ~7GB of
# VRAM instead of ~24, and it is 5.4GB in the image instead of 18.7. The acceptance target
# is a 30-minute multi-speaker file in one pass, which sits comfortably inside the 1.5B and
# would run the 7B near its ceiling. 7B stays a runtime opt-in -- see VIBEVOICE_EXTRA.
# HuggingFace 429s these downloads from GitHub's runner IPs, and `comfy model download`
# gives up after three attempts about two seconds apart -- which is no retry at all for a
# rate limiter that thinks in minutes. It killed a build here, and 0.3 records the same
# limiter stopping the LTX-2.5 pull. Several lanes in this repo also build concurrently and
# share those IPs, so the limit is hit far more often than a single image would suggest.
#
# So: retry with real backoff -- 30s, 60s, 120s, 240s, 480s. Sixteen minutes of patience
# beats a failed build and a manual re-run, and costs nothing when the first attempt works.
# Each file is verified non-trivial before being accepted, because a 429 body is a few
# hundred bytes that would otherwise sit on disk wearing a .safetensors name and only fail
# much later, at load time, on a billing GPU.
# AUTHENTICATED when the CI secret is present. Anonymous pulls from pooled GitHub
# runner IPs hit HTTP 429 once several weight-baking lanes build concurrently -- this
# lane hit it three times. The token arrives as a BuildKit secret, not a build-arg, so
# it never lands in the image layer history; `|| true` keeps a local build working with
# no secret, falling back to anonymous. The backoff below stays regardless: a higher
# rate limit is not an absent one.
ARG VV=https://huggingface.co/microsoft/VibeVoice-1.5B/resolve/main
RUN --mount=type=secret,id=hf_token \
    export HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" ; \
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" ; \
    for f in config.json model.safetensors.index.json preprocessor_config.json \
             model-00001-of-00003.safetensors \
             model-00002-of-00003.safetensors \
             model-00003-of-00003.safetensors ; do \
      ok=0 ; \
      for delay in 0 30 60 120 240 480 ; do \
        [ $delay -gt 0 ] && { echo "retrying $f in ${delay}s" ; sleep $delay ; } ; \
        if comfy model download --url "$VV/$f" \
             --relative-path models/vibevoice/VibeVoice-1.5B --filename "$f" ; then \
          ok=1 ; break ; \
        fi ; \
      done ; \
      [ $ok -eq 1 ] || { echo "FAILED after backoff: $f" ; exit 1 ; } ; \
    done

# THE TRAP. microsoft/VibeVoice-1.5B ships NO tokenizer -- look at its file list, there is
# no tokenizer.json in it. The node expects a Qwen2.5-1.5B tokenizer sitting beside the
# weights, and without it the node loads, the graph validates, and generation dies at the
# first token with a message about a missing processor. 11MB, Apache-2.0, ungated.
ARG QW=https://huggingface.co/Qwen/Qwen2.5-1.5B/resolve/main
RUN --mount=type=secret,id=hf_token \
    export HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" ; \
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" ; \
    for f in tokenizer.json tokenizer_config.json vocab.json merges.txt ; do \
      ok=0 ; \
      for delay in 0 30 60 120 240 ; do \
        [ $delay -gt 0 ] && { echo "retrying $f in ${delay}s" ; sleep $delay ; } ; \
        if comfy model download --url "$QW/$f" \
             --relative-path models/vibevoice/tokenizer --filename "$f" ; then \
          ok=1 ; break ; \
        fi ; \
      done ; \
      [ $ok -eq 1 ] || { echo "FAILED after backoff: $f" ; exit 1 ; } ; \
    done

# LoRA drop-point. The node reads adapters from models/vibevoice/loras/ and refuses to mix
# hidden sizes (a 1.5B LoRA on the 7B errors, correctly). Left EMPTY on purpose: every
# community VibeVoice LoRA found on HF as of 2026-09-12 -- Elise, Jenny, the Hindi and
# Hungarian sets -- carries NO licence tag at all, and an untagged HF repo is not
# implicitly MIT. Train our own with voicepowered-ai/VibeVoice-finetuning (MIT) instead.
RUN mkdir -p /comfyui/models/vibevoice/loras

# ---------------------------------------------------------------------------------------
# comfy-node-install did NOT install comfyui-kokoro's own requirements -- the first build
# died on `ModuleNotFoundError: No module named 'kokoro_onnx'`. That is not a build-script
# problem to work around: the node imports kokoro_onnx at module scope, so without these
# the node would fail to register at RUNTIME too, and the only symptom would be a workflow
# referencing a node class that does not exist, on a billing GPU.
#
# Pin torch first. onnxruntime and kokoro-onnx are innocuous, but the house rule exists
# because a transitive dep satisfying a torch requirement with a PyPI CUDA 13 wheel needs
# driver >= 580 while Runpod hosts run 570/575, and CUDA then dies before a single log
# line. --only-binary because source builds are what killed the sibling ../tts worker five
# times. onnxruntime (CPU) not onnxruntime-gpu: Kokoro is 82M params and CPU-fast, and the
# GPU build drags in a second CUDA runtime that can collide with torch's.
RUN python -c "import torch, torchvision, torchaudio; \
open('/tmp/constraints.txt','w').write(''.join(f'{n}=={v}\n' for n,v in \
[('torch',torch.__version__),('torchvision',torchvision.__version__),('torchaudio',torchaudio.__version__)]))" \
    && cat /tmp/constraints.txt \
    && pip install --no-cache-dir -c /tmp/constraints.txt --only-binary=:all: \
         "kokoro-onnx" "onnxruntime" \
    && python -c "import torch; v=torch.__version__; print('torch', v); \
assert 'cu13' not in v, f'torch was replaced: {v}'" \
    && python -c "import kokoro_onnx, onnxruntime; print('kokoro_onnx + onnxruntime import OK')"

# Kokoro. Two artefacts: an ONNX graph from a GitHub release, and a voice pack.
#
# WHY NOT THE NODE'S OWN DOWNLOADER. The first build used it and failed on a HuggingFace
# **429 Too Many Requests**. download_voices() pulls ~50 individual .pt files from
# hexgrad/Kokoro-82M one at a time and np.savez-es them into voices_v1.bin -- 50 sequential
# requests against a rate limiter this project has already tripped once (DECISIONS 0.3, the
# LTX-2.5 download). That is not a flake to retry; at 50 requests per build it is the
# expected outcome.
#
# So take the pre-packed voices file from the kokoro-onnx release instead: ONE GitHub
# download, no HuggingFace involved. It is the same artefact -- an npz keyed by voice name,
# exactly what np.savez produces and what kokoro_onnx.Kokoro() expects -- so the node cannot
# tell the difference. Verified before committing: 54 voices, 28.2MB, af_heart and ff_siwis
# both present with shape (510, 1, 256).
#
# The node module is still imported for its constants rather than hard-coding filenames, so
# a rename upstream surfaces here instead of as a file the node cannot find.
RUN python /dev/stdin <<'PY'
import importlib.util, os, sys, urllib.request
import numpy as np

VOICES_URL = ("https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
              "model-files-v1.0/voices-v1.0.bin")

root = "/comfyui/custom_nodes"
cands = [d for d in os.listdir(root) if "kokoro" in d.lower()]
assert cands, "kokoro node not found after comfy-node-install"
node_dir = os.path.join(root, cands[0])
spec = importlib.util.spec_from_file_location("kk", os.path.join(node_dir, "ComfyUIKokoro.py"))
mod = importlib.util.module_from_spec(spec)
sys.modules["kk"] = mod
spec.loader.exec_module(mod)

onnx = os.path.join(node_dir, mod.MODEL_FILENAME)
print("[kokoro] fetching", mod.MODEL_URL, flush=True)
urllib.request.urlretrieve(mod.MODEL_URL, onnx)
assert os.path.getsize(onnx) > 50_000_000, "onnx too small -- fetch returned an error page"
print("[kokoro] onnx", round(os.path.getsize(onnx) / 1e6, 1), "MB", flush=True)

vb = os.path.join(node_dir, mod.VOICES_FILENAME)
print("[kokoro] fetching voice pack", VOICES_URL, flush=True)
urllib.request.urlretrieve(VOICES_URL, vb)
assert os.path.getsize(vb) > 10_000_000, "voice pack too small -- fetch returned an error page"

# Prove it is a real npz containing the voices the node offers in its dropdown, rather than
# trusting the file size. A pack missing a voice the node lists would fail at request time
# on a billing GPU, naming a voice the operator picked from a menu.
have = set(np.load(vb).files)
missing = [v for v in mod.supported_voices if v not in have]
assert not missing, f"voice pack is missing voices the node offers: {missing}"
print("[kokoro] voices", round(os.path.getsize(vb) / 1e6, 1), "MB,",
      len(have), "in pack,", len(mod.supported_voices), "offered by the node", flush=True)
PY

# ---------------------------------------------------------------------------------------
# Teach the handler to return audio at all. worker-comfyui 5.10.0 walks only the "images"
# output key; SaveAudio* nodes emit under "audio", so without this every TTS job returns
# SUCCESS with an empty payload. Full reasoning in the patch script.
COPY patch_audio_outputs.py /patch_audio_outputs.py
RUN python /patch_audio_outputs.py \
    && python -c "import ast; ast.parse(open('/handler.py').read()); print('handler parses OK')" \
    && grep -q 'audio passthrough' /handler.py

COPY workflows /workflows

# ---------------------------------------------------------------------------------------
# Build-time checks that fail the BUILD rather than a billing GPU. Each of these is a real
# failure mode on this stack: a node that installs but does not import, a model whose
# download 302'd to an HTML error page and landed as a 4KB "safetensors", a missing
# tokenizer that only surfaces at the first generated token.
RUN test -s /comfyui/models/vibevoice/tokenizer/tokenizer.json && python /dev/stdin <<'PY'
import glob, json, os
d = "/comfyui/models/vibevoice/VibeVoice-1.5B"
shards = sorted(glob.glob(os.path.join(d, "*.safetensors")))
assert len(shards) == 3, f"expected 3 shards, found {shards}"
total = sum(os.path.getsize(s) for s in shards)
assert total > 5e9, f"weights only {total/1e9:.2f}GB -- a download returned an error page"
for f in ("config.json", "model.safetensors.index.json"):
    json.load(open(os.path.join(d, f)))          # HTML masquerading as JSON dies here
print(f"[check] VibeVoice-1.5B {total/1e9:.2f}GB across {len(shards)} shards, configs parse")
PY

# Boot the whole graph on CPU and exit. A broken node install, a bad pin or a missing model
# surfaces here where the log is readable, instead of as a crash-loop with no container
# output.
RUN cd /comfyui && timeout 900 python main.py --quick-test-for-ci --cpu \
    && echo "comfyui boots clean"

# Then confirm both node packs actually registered. comfy-node-install can succeed while
# the node fails to import, and the only symptom is the workflow JSON referencing a node
# class that does not exist -- at request time, on a billing GPU.
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
vv = sorted(n for n in have if "vibevoice" in n.lower())
kk = sorted(n for n in have if "kokoro" in n.lower())
assert vv, f"VibeVoice nodes missing. registered sample: {sorted(have)[:40]}"
assert kk, f"kokoro nodes missing. registered sample: {sorted(have)[:40]}"
assert any("Audio" in n for n in have), "no SaveAudio-class node registered"
print("[boot] VibeVoice:", vv, flush=True)
print("[boot] kokoro:", kk, flush=True)
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

# 7B / Large is a runtime opt-in, not a baked 18.7GB. Set VIBEVOICE_EXTRA to an HF repo id
# -- aoi-ot/VibeVoice-Large (MIT, ~18.7GB, wants 24GB VRAM) or FabioSarracino/
# VibeVoice-Large-Q8 (MIT, ~11.6GB, ~12GB VRAM) -- and point HF_HOME at a network volume so
# the pull happens once rather than on every cold start.
ENV VIBEVOICE_EXTRA="" \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1
