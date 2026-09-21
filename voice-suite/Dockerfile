# Voice suite worker: TTS-Audio-Suite, isolated on purpose.
#
# WHY ITS OWN IMAGE, AND NOT A LINE IN ../voice-comfy.
#
# TTS-Audio-Suite consolidates most of a voice lane in one pack -- Chatterbox
# (classic and multilingual), Qwen3-TTS, VibeVoice, IndexTTS-2, F5-TTS, Higgs
# Audio, RVC conversion -- plus SRT timing, which is the reason to want it: it
# aligns a read to subtitle timings instead of hoping the length matches.
#
# It also pins transformers>=5.3.0,<6.0.0, and its install.py shells out to pip
# with --force-reinstall --no-deps. ../voice-comfy runs VibeVoice and Kokoro on
# a dependency set its own Dockerfile describes as "diffusers + transformers +
# bitsandbytes + peft, all pinned against each other by the node author", and
# warns: "Extending a solved environment is the job; re-solving it is not."
# A force-reinstall into that set is the likeliest way to break a working lane,
# and it would surface on a cold start days later rather than at build time.
#
# The build matrix already states the rule this follows: each worker is a
# separate image so a dependency conflict in one cannot break the others.
#
# THE PIN IS LOAD-BEARING. install.py runs arbitrary pip at build time. That is
# acceptable HERE -- it happens once, inside an image, and the result is frozen
# and reproducible. It would not be acceptable at runtime, so nothing in this
# image should ever let ComfyUI-Manager update this pack in a live worker.
#
# Upstream is diodiogod/TTS-Audio-Suite. Note its latest git TAG is v5.6.0
# (2026-07-25) while HEAD reads "Version 5.8.11" (2026-09-19) -- it versions in
# the repo without tagging, so the tag would pin months behind. Hence a SHA.
#
# LICENCE, HONESTLY: GitHub reports NOASSERTION. The LICENSE file contains MIT
# text but is copyright "Shmuel Ronen", a different person from the maintainer
# and the second-ranked contributor -- fork lineage where the copyright line was
# never updated. Treat the grant as unsettled rather than clean MIT. Bus factor
# is 1: ~1,564 of ~1,660 commits are the maintainer's.
FROM runpod/worker-comfyui:5.10.0-base

RUN apt-get update && apt-get install -y --no-install-recommends espeak-ng ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ARG TTSAS_REF=d12a4c286f58a31f28e47a8f46e2ef7e6eb3c5c2
RUN git clone https://github.com/diodiogod/TTS-Audio-Suite.git \
        /comfyui/custom_nodes/TTS-Audio-Suite \
 && cd /comfyui/custom_nodes/TTS-Audio-Suite \
 && git checkout -q "${TTSAS_REF}" \
 && echo "TTS-Audio-Suite pinned at ${TTSAS_REF}"

# Its own installer, run ONCE here so the resolved set is baked into the image.
# Failure must stop the build: a half-installed audio stack that imports but
# cannot synthesise is worse than no image, because it only fails on a real job.
RUN cd /comfyui/custom_nodes/TTS-Audio-Suite \
 && python install.py \
 && python -c "import transformers, torch; print('transformers', transformers.__version__, '| torch', torch.__version__)"

# Prove the pack is actually there and declares nodes. ComfyUI swallows an
# import error inside a custom node at startup -- the node simply never
# registers, and the first symptom is a graph failing with an unknown
# class_type mid-render. This checks the entry point exists rather than
# guessing an importable package name.
RUN test -f /comfyui/custom_nodes/TTS-Audio-Suite/__init__.py  && grep -q "NODE_CLASS_MAPPINGS" /comfyui/custom_nodes/TTS-Audio-Suite/__init__.py  && echo "TTS-Audio-Suite declares NODE_CLASS_MAPPINGS"
