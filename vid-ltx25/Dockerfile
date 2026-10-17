# LTX-2.5 (Lightricks) video + audio worker, distilled two-stage on diffusers 0.41 (see handler.py).
# Weights are NOT in the image: they are staged on the Azure Files share by the GPU fleet's fetch job
# (local-ai-setup gpu/fetch) and mounted at /runpod-volume. The image stays small.
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# The torch280 base ships HF_HUB_ENABLE_HF_TRANSFER=1 without the package (see vid-wan-a14b). Offline on purpose:
# the weights are local, and a lane must never start pulling 100+ GB from the Hub on a billed GPU.
ENV HF_HUB_ENABLE_HF_TRANSFER=0 \
    HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    MODEL_DIR="/runpod-volume/ltx25" \
    OFFLOAD="model" \
    OUTPUT_FPS=24 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --ignore-installed cryptography
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Pin the base image's torch as constraints (CUDA 13 wheels need a newer driver; see vid-wan-a14b).
RUN python -c "import torch, torchvision, torchaudio; \
open('/tmp/constraints.txt','w').write(''.join(f'{n}=={v}\n' for n,v in \
[('torch',torch.__version__),('torchvision',torchvision.__version__),('torchaudio',torchaudio.__version__)]))" \
    && cat /tmp/constraints.txt

# diffusers 0.41.0 is the first release with LTX-2.5 (LTX2Pipeline + Gemma 4 text encoder); transformers 5 carries
# gemma4_unified; av (PyAV) is what diffusers' encode_video muxes the audio track with.
RUN pip install --no-cache-dir -c /tmp/constraints.txt \
      "diffusers==0.41.0" \
      "transformers==5.20.1" \
      "accelerate" \
      "av==19.0.1" \
      "sentencepiece" \
      "protobuf" \
      "pillow" \
      "runpod>=1.10.1"

# Fail the BUILD, not a billed GPU, if torch was swapped or the LTX-2 classes are missing.
RUN python -c "import torch; v=torch.__version__; print('torch', v); assert 'cu13' not in v, f'torch replaced: {v}'"
RUN python -c "\
import diffusers, transformers, av; \
from diffusers import LTX2Pipeline, LTX2ConditionPipeline; \
from diffusers.pipelines.ltx2 import LTX2LatentUpsamplePipeline; \
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel; \
from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition; \
from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES; \
from diffusers.utils import encode_video; \
from transformers.models.gemma4_unified import configuration_gemma4_unified; \
print('diffusers', diffusers.__version__, '| transformers', transformers.__version__, '| av', av.__version__, '| LTX-2.5 classes OK')"

COPY handler.py /handler.py
RUN python -c "import ast; ast.parse(open('/handler.py').read()); print('handler parses OK')"

CMD ["python", "-u", "/handler.py"]
