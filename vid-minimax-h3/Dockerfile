# MiniMax-H3 (MiniMaxAI/MiniMax-H3, 144GB): text/first-frame-and-last-frame to video WITH audio, joint
# denoise, no separate vocoder. Azure GPU fleet lane `minimax-h3`, Canada Central only -- see the LICENCE
# block below and README.md before touching anything in this file.
#
# BUILD AND PUSH FROM CANADA ONLY. `azswgpuacrca` (Canada Central), never `azswmcacr` (East US 2): the
# MiniMax H3 Community License's Territory excludes the EU, UK, South Korea AND the United States, and it
# binds the Works (this image, once it holds no weights, is fine; but never give it the chance to be built
# or stored on a US registry out of habit -- see every other video lane in this repo, which IS allowed in
# the US and does build on azswmcacr).
#
# Weights are NOT in the image: they are staged on the Canada Azure Files share by the fetch job
# (local-ai-setup gpu/manifests/minimax-h3.tsv) and mounted at /runpod-volume, same flat-directory
# convention as every other lane here (vid-hunyuan-i2v, vid-wan-animate, vid-ltx25).
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# The torch280 base ships HF_HUB_ENABLE_HF_TRANSFER=1 without the hf_transfer package (see vid-hunyuan-i2v
# for the first lane this bit). HF_HUB_OFFLINE=1 means nothing downloads at runtime regardless; turning
# hf_transfer off too costs nothing and removes one thing that could go wrong first.
ENV HF_HUB_ENABLE_HF_TRANSFER=0 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    MINIMAX_H3_DIR="/runpod-volume/minimax-h3" \
    MEMORY_RESERVE_MARGIN="12GB" \
    OUTPUT_FPS=24 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --ignore-installed cryptography
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Pin the base image's torch as constraints (see vid-wan-a14b/Dockerfile for the full story): left
# unpinned, diffusers/accelerate's bare "torch" dependency resolves to a CUDA 13 wheel from PyPI, which
# needs driver >= 580 and crash-loops with an empty log before Python even starts.
RUN python -c "import torch, torchvision, torchaudio; \
open('/tmp/constraints.txt','w').write(''.join(f'{n}=={v}\n' for n,v in \
[('torch',torch.__version__),('torchvision',torchvision.__version__),('torchaudio',torchaudio.__version__)]))" \
    && cat /tmp/constraints.txt

# diffusers 0.41.0 is the first PyPI release carrying MiniMax-H3 as Modular Diffusers blocks (added to
# diffusers main 2026-08-05 PR #14355, with a VAE-decode fix 2026-09-28 #14754 and single-file support
# 2026-10-01 #14839 -- 0.41.0 postdates all three, verified against the published wheel rather than
# installed from git at a pinned commit, which the model's own upstream doc still recommends for a repo
# this new). This exact diffusers==0.41.0 / transformers==5.20.1 pair is already proven in this repo on
# vid-ltx25 (LTX-2.5, also diffusers 0.41's launch model) -- reused rather than re-derived.
# transformers>=4.57.0 is MiniMax-H3's own floor (Qwen3VLForConditionalGeneration did not exist before
# it); 5.20.1 is current and already the pin on vid-ltx25 in this same repo.
# av (PyAV): diffusers' encode_video muxes the generated audio track through it.
RUN pip install --no-cache-dir -c /tmp/constraints.txt \
      "diffusers==0.41.0" \
      "transformers==5.20.1" \
      "accelerate" \
      "av==19.0.1" \
      "sentencepiece" \
      "protobuf" \
      "pillow" \
      "runpod>=1.10.1"

# safe_fetch.py: the SSRF guard for any job input that is an https:// URL rather than inline data (job
# inputs can arrive from an LLM agent, not just a human -- see its own module docstring). Copied in before
# build_check.py so build_check's own safe_fetch subcommand can import it. Copied VERBATIM from
# vid-hunyuan-i2v -- same file, same guarantees, no shared base layer between the two workers.
COPY safe_fetch.py /safe_fetch.py

# Fail the BUILD, not a billed GPU: a file, not RUN python -c "..." continuation lines -- ACR's
# dependency scanner reads any line starting with `from` as a FROM instruction (vid-wan-a14b, 2026-10-17).
COPY build_check.py /build_check.py
RUN python /build_check.py imports
RUN python /build_check.py offline
RUN python /build_check.py safe_fetch

COPY handler.py /handler.py
RUN python -c "import ast; ast.parse(open('/handler.py').read()); print('handler parses OK')"

CMD ["python", "-u", "/handler.py"]
