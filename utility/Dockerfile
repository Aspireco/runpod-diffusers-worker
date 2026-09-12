# BiRefNet cutouts. ~450MB of weights, fetched on first boot.
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf \
    MODEL_ID=ZhengPeng7/BiRefNet \
    INPUT_RES=1024 \
    HF_HUB_DISABLE_TELEMETRY=1

RUN pip install --no-cache-dir --ignore-installed cryptography
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# BiRefNet loads through trust_remote_code, which pulls in timm and einops.
# torch/torchvision are left as the base image ships them -- reinstalling pulls CUDA 13
# wheels needing driver >= 580 while Runpod hosts run 570/575.
RUN pip install --no-cache-dir \
      "transformers>=4.50.3,<5" \
      "huggingface-hub<1.0" \
      "timm" \
      "einops" \
      "kornia" \
      "runpod>=1.10.1"

COPY handler.py /handler.py
RUN python -c "import ast; ast.parse(open('/handler.py').read()); print('handler parses OK')"

CMD ["python", "-u", "/handler.py"]
