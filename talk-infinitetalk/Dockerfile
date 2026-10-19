# InfiniteTalk talking presenter (Apache-2.0, MeiGen-AI). A photo or video + an audio track -> a
# lip-synced talking video. Weights come from the Azure Files share (/runpod-volume/infinitetalk/...),
# staged by the fetch jobs for gpu/manifests/infinitetalk-*.tsv; the image holds none of them.
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV HF_HUB_ENABLE_HF_TRANSFER=0 \
    HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/infinitetalk \
    INFINITETALK_DIR=/runpod-volume/infinitetalk \
    LIGHTNING=1 \
    OFFLOAD_MODEL=0

# ffmpeg: muxing, the h264 re-encode of re-dub inputs, and the official code's own av1->h264 fallback.
# libsndfile1: soundfile (librosa's wav backend). git: clones the pinned InfiniteTalk commit below.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --ignore-installed cryptography
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Pin the base image's own torch as constraints (CUDA 13 wheels need driver >= 580; see vid-wan-a14b).
RUN python -c "import torch, torchvision, torchaudio; \
open('/tmp/constraints.txt','w').write(''.join(f'{n}=={v}\n' for n,v in \
[('torch',torch.__version__),('torchvision',torchvision.__version__),('torchaudio',torchaudio.__version__)]))" \
    && cat /tmp/constraints.txt

# Official code (Apache-2.0), pinned to the commit this worker was built and tested against (see README.md
# "Premade search"). Only wan/ (the generation pipeline) and src/ (audio encoder + VRAM management) are
# kept: kokoro/ (TTS -- unused, a job always supplies an explicit audio track), tools/, app.py (gradio demo),
# examples/ and assets/ (sample media + the paper PDF) are dropped to keep the image lean. LICENSE.txt is
# kept for the record.
ENV INFINITETALK_COMMIT=50aa0a94184315407a991ae804d9b58d6d311ba8
RUN git clone --quiet https://github.com/MeiGen-AI/InfiniteTalk.git /opt/infinitetalk_src \
    && cd /opt/infinitetalk_src && git checkout --quiet ${INFINITETALK_COMMIT} \
    && mkdir -p /opt/infinitetalk \
    && mv wan src LICENSE.txt /opt/infinitetalk/ \
    && cd / && rm -rf /opt/infinitetalk_src

# One-line build-time compatibility patch, not a behaviour change: wan/modules/t5.py's T5EncoderModel.__init__
# default is `device=torch.cuda.current_device()`, a DEFAULT PARAMETER VALUE, so Python evaluates it the
# moment the class body executes -- i.e. at `import wan` time, before any job runs. ACR build agents have no
# GPU at all, so that raises "Found no NVIDIA driver" and failed the first build attempt here (run ch1u,
# 2026-10-17). The only caller in this worker's path, InfiniteTalkPipeline.__init__ (wan/multitalk.py), always
# passes device=torch.device('cpu') explicitly, so this default is never actually read by anything this
# handler does; replacing it with the literal `0` -- torch.cuda.current_device()'s own return type and value
# on any single-GPU machine, which is exactly this worker's target hardware -- keeps it import-safe on a
# CPU-only build machine with identical runtime behaviour on the A100.
RUN sed -i 's/device=torch\.cuda\.current_device(),/device=0,/' /opt/infinitetalk/wan/modules/t5.py \
    && grep -n "device=0," /opt/infinitetalk/wan/modules/t5.py

# A second, same-shape build-time patch (run ch1w, 2026-10-17): wan/multitalk.py has a dead, unused
# `from inspect import ArgSpec` at its top -- `ArgSpec` was removed from the stdlib `inspect` module in
# Python 3.11 (this base image ships 3.12), and grep confirms the name is never referenced anywhere else
# in the file. Not a behaviour change: it deletes an import nothing calls.
RUN sed -i '/^from inspect import ArgSpec$/d' /opt/infinitetalk/wan/multitalk.py \
    && ! grep -n "ArgSpec" /opt/infinitetalk/wan/multitalk.py

# xformers is a hard, unconditional import for the audio cross-attention path (wan/modules/attention.py
# imports xformers.ops with no try/except, used by SingleStreamAttention.forward).
#
# flash-attn IS required. attention() in wan/modules/attention.py does fall back to torch's SDPA, but Wan's CLIP
# encoder (wan/modules/clip.py) and the DiT itself (wan/modules/multitalk_model.py, unless sageattention is
# installed) call flash_attention() directly, and that function does a bare `assert FLASH_ATTN_2_AVAILABLE`. The
# image without it failed its second Azure proof on an empty AssertionError right after the CLIP encode
# (2026-10-18). The wheel is Dao-AILab's prebuilt one (BSD-3-Clause), matched to this base's torch 2.8 / CUDA 12 /
# Python 3.12, with the C++ ABI read from the installed torch -- no compile on ACR. 2.8.1, not 2.8.3: xformers
# 0.0.32.post2 refuses flash-attn above 2.8.2 (the build check caught it, ACR run ch3g).
RUN ABI=$(python -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')") \
    && pip install --no-cache-dir --no-deps \
      "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.8cxx11abi${ABI}-cp312-cp312-linux_x86_64.whl"
#
# xformers==0.0.32.post2 is the release whose own requirements.txt pins `torch >= 2.8` (checked at tag
# v0.0.32.post2) and ships a cp39-abi3 manylinux wheel (no compile, works on any Python >= 3.9).
#
# xfuser is imported unconditionally too (wan/modules/attention.py, for get_sequence_parallel_rank /
# get_sequence_parallel_world_size / get_sp_group) even though this worker never uses sequence-parallel
# (single GPU, ulysses_size=1). Its own hard dependency (yunchang) makes flash-attn an opt-in "extra",
# never a default install, so plain `pip install xfuser` does not pull it either (checked on PyPI).
# Pinned to 0.4.5 -- the last 0.4.x release, the same minor line as the InfiniteTalk repo's own
# `xfuser>=0.4.1` floor -- rather than jumping to 0.5-0.7 (2026 releases, after this repo went quiet).
#
# optimum-quanto==0.2.6 matches the official requirements.txt exactly: multitalk.py imports its qint8/
# requantize/qlinear internals directly, which is the kind of API that can rename across versions.
#
# opencv-python-headless replaces upstream's opencv-python: same `cv2` import, no libGL/libglib needed in
# a GPU container with no display. numpy is capped <2 because the official requirements.txt caps it there.
#
# einops, regex, librosa, soundfile and safetensors are genuinely imported by the vendored code (grep-
# verified file by file) but are MISSING from the official requirements.txt -- added explicitly here
# rather than relying on them arriving as someone else's transitive dependency.
RUN pip install --no-cache-dir -c /tmp/constraints.txt \
      "xformers==0.0.32.post2" \
      "xfuser==0.4.5" \
      "diffusers>=0.31.0" \
      "transformers>=4.49.0,<5" \
      "tokenizers>=0.20.3" \
      "accelerate>=1.1.1" \
      "optimum-quanto==0.2.6" \
      "decord" \
      "librosa" \
      "soundfile" \
      "pyloudnorm" \
      "einops" \
      "regex" \
      "safetensors" \
      "easydict" \
      "ftfy" \
      "scikit-image" \
      "loguru" \
      "opencv-python-headless>=4.9.0.80" \
      "imageio" \
      "imageio-ffmpeg" \
      "numpy>=1.23.5,<2" \
      "tqdm" \
      "runpod>=1.10.1"

COPY build_check.py /build_check.py
RUN python /build_check.py
COPY handler.py /handler.py
RUN python -c "import ast; ast.parse(open('/handler.py').read()); print('handler parses OK')"

CMD ["python", "-u", "/handler.py"]
