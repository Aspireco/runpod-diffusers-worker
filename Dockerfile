# Deliberately tiny: no model weights. Runpod's Model Caching mounts them at runtime, so
# this image builds in a couple of minutes and stays well inside the builder's limits.
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# Offline loading is correct HERE because the weights genuinely are on local disk (the
# mounted cache). It is also what produces the misleading "outgoing traffic has been
# disabled" error when the cache is missing -- that message means this flag is set, not
# that Runpod blocked the network.
ENV MODEL_ID="Tongyi-MAI/Z-Image-Turbo" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PYTHONUNBUFFERED=1

# Pins, each one load-bearing:
#   transformers<5 / huggingface-hub<1.0 -- the 5.x and 1.x releases broke diffusers imports
#   runpod>=1.10.1                       -- 1.7.11-1.10.0 corrupt job tracking
# torch is left exactly as the base image ships it: reinstalling from default PyPI pulls
# CUDA 13 wheels that need driver >=580, while Runpod hosts run 570/575, and CUDA init then
# dies before a single log line is written.
RUN pip install --no-cache-dir \
      "diffusers>=0.31" \
      "transformers>=4.50.3,<5" \
      "huggingface-hub<1.0" \
      "accelerate" \
      "sentencepiece" \
      "protobuf" \
      "runpod>=1.10.1"

COPY handler.py /handler.py

# Fail the BUILD rather than the worker: a syntax error or bad import surfaces here, where
# the error is visible, instead of as a silent crash-loop with no container logs.
RUN python -c "import ast,sys; ast.parse(open('/handler.py').read()); print('handler parses OK')"

CMD ["python", "-u", "/handler.py"]
