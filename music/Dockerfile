# ACE-Step music generation. Weights download on first boot into ACE_CHECKPOINT_DIR.
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf \
    ACE_CHECKPOINT_DIR=/opt/ace-checkpoints \
    HF_HUB_DISABLE_TELEMETRY=1

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --ignore-installed cryptography
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# ACE-Step is not published to PyPI — "pip install acestep" finds no distribution at all.
# It installs from source. Pinned to a commit-less main because the project tags rarely.
RUN pip install --no-cache-dir "numpy>=1.26,<3"
RUN pip install --no-cache-dir --no-build-isolation \
      "git+https://github.com/ace-step/ACE-Step.git" \
    && pip install --no-cache-dir \
      "transformers>=4.50.3,<5" \
      "huggingface-hub<1.0" \
      "runpod>=1.10.1"

RUN mkdir -p /opt/ace-checkpoints

COPY handler.py /handler.py
RUN python -c "import ast; ast.parse(open('/handler.py').read()); print('handler parses OK')"

CMD ["python", "-u", "/handler.py"]
