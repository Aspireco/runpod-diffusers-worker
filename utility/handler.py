"""Cutout worker: BiRefNet (MIT) background removal with a real alpha channel.

This is the highest-frequency operation in product work -- every catalogue shot, every
composite, every ad variant starts with a clean cutout. BiRefNet is here rather than the
better-known RMBG-2.0 because RMBG is CC-BY-NC and needs a paid BRIA agreement.

MODEL_ID picks the variant:
  ZhengPeng7/BiRefNet          general purpose, 1024px
  ZhengPeng7/BiRefNet_HR       2048px, for catalogue work that gets zoomed
  egeorcun/lucida              BiRefNet fine-tune for glass, gloss and logos -- the one
                               for stainless food equipment and display cases
"""

import sys

# First line out, before any heavy import. A crash-looping worker with NO container logs
# at all means the process died before this ran (bad image, bad CUDA); one that prints this
# and then stops died during an import, which is a completely different fix. Without it the
# two are indistinguishable and you end up guessing.
print("[boot] handler starting", flush=True)

import base64
import io
import os
import time
import urllib.request

import torch
print(f"[boot] torch {torch.__version__} cuda={torch.cuda.is_available()}", flush=True)

from PIL import Image

MODEL_ID = os.environ.get("MODEL_ID", "ZhengPeng7/BiRefNet")
RES = int(os.environ.get("INPUT_RES", "1024"))

_t0 = time.time()
from transformers import AutoModelForImageSegmentation  # noqa: E402

MODEL = AutoModelForImageSegmentation.from_pretrained(MODEL_ID, trust_remote_code=True)
MODEL.to("cuda").eval()
MODEL.half()
torch.set_float32_matmul_precision("high")
print(f"[boot] {MODEL_ID} ready in {time.time() - _t0:.1f}s", flush=True)

from torchvision import transforms  # noqa: E402

PREP = transforms.Compose([
    transforms.Resize((RES, RES)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _load_image(job_input):
    src = job_input.get("image_url") or job_input.get("image_base64") or job_input.get("image")
    if not src:
        return None
    if src.startswith("http"):
        with urllib.request.urlopen(src, timeout=60) as r:
            data = r.read()
    else:
        raw = src.split(",", 1)[1] if src.startswith("data:") else src
        data = base64.b64decode(raw)
    return Image.open(io.BytesIO(data)).convert("RGB")


def handler(job):
    job_input = job.get("input") or {}
    started = time.time()
    try:
        img = _load_image(job_input)
        if img is None:
            return {"error": "image_url or image_base64 is required"}

        w, h = img.size
        x = PREP(img).unsqueeze(0).to("cuda").half()

        with torch.no_grad():
            preds = MODEL(x)[-1].sigmoid().cpu()

        mask = transforms.ToPILImage()(preds[0].squeeze()).resize((w, h))

        # Composite onto the requested background. Transparent is the useful default for
        # downstream compositing; a solid colour saves a step for catalogue listings.
        bg = (job_input.get("background") or "transparent").lower()
        if bg == "transparent":
            out = img.copy().convert("RGBA")
            out.putalpha(mask)
            fmt = "PNG"
        else:
            colour = (255, 255, 255) if bg == "white" else (0, 0, 0) if bg == "black" else None
            if colour is None:
                colour = tuple(int(bg.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
            canvas = Image.new("RGB", (w, h), colour)
            canvas.paste(img, mask=mask)
            out = canvas
            fmt = "PNG"

        buf = io.BytesIO()
        out.save(buf, format=fmt)
        result = {
            "image_url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
            "width": w, "height": h, "model": MODEL_ID,
            "generation_seconds": round(time.time() - started, 2),
        }

        if job_input.get("return_mask"):
            mbuf = io.BytesIO()
            mask.save(mbuf, format="PNG")
            result["mask_url"] = "data:image/png;base64," + base64.b64encode(mbuf.getvalue()).decode()
        return result
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


import runpod  # noqa: E402

runpod.serverless.start({"handler": handler})
