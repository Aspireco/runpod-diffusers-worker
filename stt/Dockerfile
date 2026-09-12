# faster-whisper transcription. large-v3 at int8 is ~1.5GB, so it downloads on first boot
# in seconds rather than being baked in.
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf \
    WHISPER_MODEL=large-v3 \
    WHISPER_COMPUTE=int8_float16 \
    HF_HUB_DISABLE_TELEMETRY=1

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --ignore-installed cryptography
RUN pip install --no-cache-dir \
      "faster-whisper" \
      "huggingface-hub<1.0" \
      "runpod>=1.10.1"

COPY handler.py /handler.py
RUN python -c "import ast; ast.parse(open('/handler.py').read()); print('handler parses OK')"

CMD ["python", "-u", "/handler.py"]
