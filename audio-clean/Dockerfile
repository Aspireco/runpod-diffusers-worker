# Audio cleanup (ElevenLabs "Voice Isolator" equivalent) for the Azure GPU fleet lane `audio-clean`
# (T4, 16 GB). Demucs v4 (MIT, facebookresearch/demucs) separates a mix into stems; DeepFilterNet 3
# (MIT OR Apache-2.0, Rikorose/DeepFilterNet) denoises speech. See README.md for the full contract,
# the licence table and the "Premade search" notes.
#
#   mode=isolate_voice (default): Demucs "vocals" stem -> DeepFilterNet denoise
#   mode=remove_voice:            Demucs "drums"+"bass"+"other" stems summed (the music/ambience bed)
#   mode=denoise:                 DeepFilterNet only
#
# Both sets of weights are baked in below: there is no fetch job for this lane, and the combined total
# (measured for real on ACR run ch2f, 2026-10-17: 429,423,306 B = 429.4 MB, both Demucs variants +
# DeepFilterNet3) is comfortably under the task's ~500MB bake allowance -- checked for real at the end of
# this file, not just asserted here, so a future upstream release silently growing these files fails the
# BUILD.
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV HF_HUB_ENABLE_HF_TRANSFER=0 \
    HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PYTHONUNBUFFERED=1 \
    DEMUCS_DIR=/opt/weights/demucs \
    DEMUCS_MODEL=htdemucs \
    DFN_DIR=/opt/weights/DeepFilterNet3

# ffmpeg: all audio/video decode, resample and mux (system build, checked for libx264 below -- a conda
# ffmpeg without it broke another worker; this one only ever needs "copy" + the built-in aac encoder on
# its video path, but the check is cheap and matches this repo's standing policy for any video-touching
# lane). libsndfile1: torchaudio/soundfile's wav backend.
#
# build-essential + cargo + rustc: deepfilterlib (the Rust DSP core DeepFilterNet's `df` package binds
# to via PyO3) has never shipped a cp312 wheel on PyPI, in any release up to and including the current
# latest 0.5.6 (checked against PyPI's own release JSON 2026-10-17: wheels stop at cp311; the upstream
# repo's last release was 2024-10-17, before cp312 wheels were the norm for small PyO3 crates) -- and
# this base image ships Python 3.12. `pip install deepfilternet` therefore builds deepfilterlib from its
# sdist via maturin, which needs a working `cargo`/`rustc` *and* a host C linker (the lesson from
# lipsync-musetalk's own Dockerfile: "the base image's 'devel' CUDA layer does not itself guarantee a
# host gcc" -- true here too, hence build-essential). deepfilterlib's own Cargo.toml pins `edition =
# "2021"` and `pyo3 = "0.19"`, both well inside what Ubuntu 24.04's packaged cargo (1.75.0, checked
# against packages.ubuntu.com 2026-10-17) supports -- no reason to reach for rustup instead.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg libsndfile1 build-essential cargo rustc curl unzip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --ignore-installed cryptography
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Pin the base image's own torch as constraints (CUDA 13 wheels need a newer driver; see voice-qwen3).
RUN python -c "import torch, torchvision, torchaudio; \
open('/tmp/constraints.txt','w').write(''.join(f'{n}=={v}\n' for n,v in \
[('torch',torch.__version__),('torchvision',torchvision.__version__),('torchaudio',torchaudio.__version__)]))" \
    && cat /tmp/constraints.txt

# demucs==4.1.0's own requires_dist (checked at source via PyPI's JSON, 2026-10-17): einops,
# huggingface-hub, julius, lameenc, pyyaml, safetensors, sphn, torch>=2.1, tqdm -- all already satisfied
# or pulled in cleanly, none pin numpy or conflict with the base image's torch 2.8 on linux.
#
# dora-search is added explicitly even though PyPI gates it behind demucs's own "train" extra: the
# installed demucs/pretrained.py has an UNCONDITIONAL `from dora.log import fatal, bold` at module level
# (confirmed by reading the file directly, 2026-10-17) that a plain `pip install demucs` would otherwise
# leave unmet -- build_check.py below proves the actual import+load chain works, not just this pin.
# dora-search is MIT; its own deps (omegaconf BSD, retrying Apache-2.0, submitit MIT, treetable Unlicense)
# checked individually -- no GPL/AGPL anywhere in this chain.
#
# deepfilternet==0.5.6 hard-pins deepfilterlib==0.5.6 itself (its own requires_dist); appdirs/loguru/
# numpy/packaging/requests/sympy come along as its other direct, non-extra deps.
RUN pip install --no-cache-dir -c /tmp/constraints.txt \
      "demucs==4.1.0" \
      "dora-search==0.1.13" \
      "deepfilternet==0.5.6" \
      "soundfile" \
      "runpod>=1.10.1"

# ---- Demucs weights: htdemucs (single model, the default -- one forward pass) + htdemucs_ft (bag of 4
# fine-tuned sources, ~4x the compute, opt in via DEMUCS_MODEL=htdemucs_ft) -- straight from Meta's own
# release bucket, laid out exactly as demucs's own `--repo <dir>` local-repo format expects (demucs/
# repo.py: one "<sig>-<checksum>.th" file per signature, plus a "<name>.yaml" bag file per name -- the
# same files demucs/remote/*.{txt,yaml} inside the pip package itself point at). Sizes are Content-Length
# from dl.fbaipublicfiles.com, checked by hand 2026-10-17, asserted below so a swapped/corrupted file
# fails the build rather than loading silently-wrong weights.
RUN mkdir -p "$DEMUCS_DIR" \
 && curl -fL --retry 3 --retry-delay 5 -o "$DEMUCS_DIR/955717e8-8726e21a.th" \
      https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th \
 && curl -fL --retry 3 --retry-delay 5 -o "$DEMUCS_DIR/f7e0c4bc-ba3fe64a.th" \
      https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/f7e0c4bc-ba3fe64a.th \
 && curl -fL --retry 3 --retry-delay 5 -o "$DEMUCS_DIR/d12395a8-e57c48e6.th" \
      https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/d12395a8-e57c48e6.th \
 && curl -fL --retry 3 --retry-delay 5 -o "$DEMUCS_DIR/92cfc3b6-ef3bcb9c.th" \
      https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/92cfc3b6-ef3bcb9c.th \
 && curl -fL --retry 3 --retry-delay 5 -o "$DEMUCS_DIR/04573f0d-f3cf25b2.th" \
      https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/04573f0d-f3cf25b2.th \
 && printf "models: ['955717e8']\n" > "$DEMUCS_DIR/htdemucs.yaml" \
 && printf "models: ['f7e0c4bc', 'd12395a8', '92cfc3b6', '04573f0d']\nweights: [\n  [1., 0., 0., 0.],\n  [0., 1., 0., 0.],\n  [0., 0., 1., 0.],\n  [0., 0., 0., 1.],\n]\n" > "$DEMUCS_DIR/htdemucs_ft.yaml" \
 && test "$(wc -c < "$DEMUCS_DIR/955717e8-8726e21a.th")" = "84141911" \
 && test "$(wc -c < "$DEMUCS_DIR/f7e0c4bc-ba3fe64a.th")" = "84141271" \
 && test "$(wc -c < "$DEMUCS_DIR/d12395a8-e57c48e6.th")" = "84141271" \
 && test "$(wc -c < "$DEMUCS_DIR/92cfc3b6-ef3bcb9c.th")" = "84141271" \
 && test "$(wc -c < "$DEMUCS_DIR/04573f0d-f3cf25b2.th")" = "84141271"

# ---- DeepFilterNet3 checkpoint (7,986,207 B zip, 8,716,140 B extracted: config.ini + checkpoints/
# model_120.ckpt.best): straight from the repo's own models/ release asset -- init_df()'s own
# download_file() would fetch this exact URL into an appdirs cache at the first job; baking it removes
# that runtime network dependency entirely. Extracted directly under /opt/weights so the zip's own
# "DeepFilterNet3/" folder name becomes $DFN_DIR, matching what a real (non-None, non-bare-name)
# `model_base_dir` argument to init_df() expects: the directory itself holding config.ini + checkpoints/.
RUN curl -fL --retry 3 --retry-delay 5 -o /tmp/dfn3.zip \
      https://github.com/Rikorose/DeepFilterNet/raw/main/models/DeepFilterNet3.zip \
 && test "$(wc -c < /tmp/dfn3.zip)" = "7986207" \
 && unzip -q /tmp/dfn3.zip -d /opt/weights \
 && rm /tmp/dfn3.zip \
 && test -f "$DFN_DIR/config.ini" \
 && test -f "$DFN_DIR/checkpoints/model_120.ckpt.best"

# Combined weight budget: fail the BUILD, not a billed GPU, if the baked weights ever exceed the task's
# ~500MB bake allowance (measured total 429,423,306 B -- see the file header).
RUN total=$(du -scb "$DEMUCS_DIR" "$DFN_DIR" | tail -1 | cut -f1) \
 && echo "baked weights: $total bytes" \
 && [ "$total" -lt 500000000 ]

COPY build_check.py /build_check.py
RUN python /build_check.py
# the optional video path muxes with "copy" (no h264 encode) but still needs the built-in aac encoder;
# this also matches the repo's standing check for any ffmpeg-touching lane, in case a future change here
# ever needs a real h264 encode. No `-q`: with this shell's `pipefail` on, `grep -q` closes its read end
# the moment it matches, SIGPIPEing ffmpeg mid-write (exit 141) before it finishes the encoder list --
# found by a real build failure here (run ch2g, 2026-10-17), not by inspection. Plain `grep` drains the
# whole pipe, so ffmpeg always exits 0, and the matched line doubles as the build-log proof.
RUN /usr/bin/ffmpeg -hide_banner -encoders 2>/dev/null | grep libx264

COPY handler.py /handler.py
RUN python -c "import ast; ast.parse(open('/handler.py').read()); print('handler parses OK')"

CMD ["python", "-u", "/handler.py"]
