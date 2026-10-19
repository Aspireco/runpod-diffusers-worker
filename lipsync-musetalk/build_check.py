"""Run at image build time: fail the BUILD, not a billed GPU (a file, because ACR's scanner
misreads Dockerfile RUN lines that begin with `from`).

Verifies every third-party package the handler imports, plus every vendored MuseTalk module the
handler calls into, with ONE deliberate exception: `musetalk.utils.preprocessing`. At module
scope that file constructs the DWPose pose model and the S3FD face detector from files under
`./models/...` -- which only exist once the fetch job has staged $MUSETALK_DIR onto the Azure
Files share, not on this GPU-less ACR build agent -- and it also wants a CUDA device, which this
build agent does not have either. handler.py's own boot sequence is where that import runs for
real, against the real files, inside a real worker replica with a real T4. Every import below was
traced line-by-line against the actual MuseTalk source (commit 0a89dec45a0192b824e3cf4daf96c239440c5ed8)
rather than guessed; see README.md.
"""
import sys

sys.path.insert(0, "/opt/MuseTalk")

# Third-party stack (the fragile part: mmcv/mmdet/mmpose prebuilt wheels pinned to torch 2.0.1/cu117).
import cv2  # noqa: E402,F401
import torch  # noqa: E402
import torchvision  # noqa: E402
import einops  # noqa: E402,F401
import omegaconf  # noqa: E402,F401
import soundfile  # noqa: E402,F401
import librosa  # noqa: E402,F401
import transformers  # noqa: E402
import diffusers  # noqa: E402
import mmcv  # noqa: E402
import mmdet  # noqa: E402
import mmpose  # noqa: E402
import mmengine  # noqa: E402,F401
from mmpose.apis import inference_topdown, init_model  # noqa: E402,F401
from mmpose.structures import merge_data_samples  # noqa: E402,F401
from transformers import WhisperModel, AutoFeatureExtractor  # noqa: E402,F401
from diffusers import AutoencoderKL, UNet2DConditionModel  # noqa: E402,F401

# Vendored MuseTalk code the handler calls into directly (class/function definitions only --
# nothing below instantiates a model or opens a weight file at import time).
import musetalk.utils.utils  # noqa: E402,F401 - load_all_model, datagen, get_video_fps
import musetalk.utils.blending  # noqa: E402,F401 - get_image
import musetalk.utils.audio_processor  # noqa: E402,F401 - AudioProcessor
import musetalk.utils.face_parsing  # noqa: E402,F401 - FaceParsing (model_init() not called here)
import musetalk.models.vae  # noqa: E402,F401
import musetalk.models.unet  # noqa: E402,F401

# The vendored S3FD/FAN face-detection package (1adrianb/face-alignment, BSD-3-Clause). This needs
# musetalk.utils already imported above -- that module's own __init__.py is what puts
# musetalk/utils/face_detection onto sys.path. Importing it only builds classes; FaceAlignment()
# (which would load the s3fd weight and wants CUDA) is never called here.
import face_detection  # noqa: E402,F401

import runpod  # noqa: E402,F401

assert "cu13" not in torch.__version__, f"torch replaced: {torch.__version__}"
assert torch.__version__.startswith("2.0.1"), f"unexpected torch: {torch.__version__}"
print("torch", torch.__version__, "| torchvision", torchvision.__version__,
      "| transformers", transformers.__version__, "| diffusers", diffusers.__version__,
      "| mmcv", mmcv.__version__, "| mmdet", mmdet.__version__, "| mmpose", mmpose.__version__,
      "| musetalk + face_detection import OK")
