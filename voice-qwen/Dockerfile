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
RUN NODE_DIR="$(find /comfyui/custom_nodes -maxdepth 1 -iname '*qwen*tts*' | head -1)" \
    && test -n "$NODE_DIR" || (echo "qwen3-tts node not installed" && exit 1) \
    && python "$NODE_DIR/download_models.py" --target /comfyui/models/qwen-tts

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
print("[boot] importing nodes", flush=True)
import nodes
nodes.init_extra_nodes(init_custom_nodes=True)
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
