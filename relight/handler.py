"""Relighting and scene compositing: IC-Light v1 on SD1.5.

The job this does: take a cut-out product and a target scene, and make the product look
like it was genuinely lit by that scene -- matched key light, matched colour temperature,
a shadow that touches the ground. Cutting out is NOT this worker's job; the BiRefNet
worker in ../utility already does that, and the caller should pass its RGBA output here.

LICENCE TRAIL (all three layers have to be commercially clean, not just the top one):
  code    lllyasviel/IC-Light                    Apache-2.0
  deltas  lllyasviel/ic-light                    CreativeML OpenRAIL-M -- commercial OK
  base    stablediffusionapi/realistic-vision-v51 CreativeML OpenRAIL-M -- commercial OK, ungated

Deliberately NOT used:
  IC-Light v2  -- FLUX-based, non-commercial licence. Do not "upgrade" to it.
  RMBG-1.4     -- IC-Light's own demo defaults to it, but it is CC-BY-NC. This worker has
                  no background remover at all; pass an RGBA cut-out instead.
  libcom       -- would have given us shadow generation and harmonisation for free, but its
                  requirements.txt hard-pins torch==2.6.0 / torchvision==0.21.0 /
                  transformers==4.44.2 and pulls the mmdet+mmpose+mmcv compile chain.
                  Downgrading torch inside this base image is the documented way to produce
                  a silent crash-loop. The shadow and detail passes below are hand-rolled
                  in numpy/PIL instead -- smaller, and they cost no extra dependency.

HOW IC-LIGHT ACTUALLY WORKS (this is not a normal diffusers pipeline):
  The released .safetensors are not a model, they are a per-tensor OFFSET added to any
  SD1.5 UNet's state dict. The UNet's first conv is widened from 4 input channels to 8 (fc)
  or 12 (fbc), the extra channels zero-initialised, and then the offset is summed in. At
  sampling time the conditioning latents are concatenated onto the noisy latent by a hook
  on unet.forward, smuggled through diffusers' cross_attention_kwargs.

  fbc (12ch) takes fg latent + bg latent  -> relights the product INTO a supplied scene.
  fc  ( 8ch) takes fg latent only         -> relights from the text prompt alone.
  Two different conv_in shapes means two different UNets; both are built at import.

  The magic number is 127. numpy2pytorch divides by 127.0 (not 127.5) so that a pixel
  value of exactly 127 maps to exactly 0.0 in the conditioning latent -- that is the
  model's "nothing here" signal. Every transparent pixel must therefore become 127-grey,
  which is why _fit_on_grey exists and why compositing a cut-out onto white or black
  quietly wrecks the output.
"""

import base64
import io
import math
import os
import random
import time
import urllib.request

import numpy as np
import torch
from PIL import Image, ImageFilter

BASE_MODEL = os.environ.get("BASE_MODEL", "stablediffusionapi/realistic-vision-v51")
IC_LIGHT_REPO = os.environ.get("IC_LIGHT_REPO", "lllyasviel/ic-light")
# The text-conditioned UNet is a second ~1.7GB download and ~1.7GB of VRAM. Worth it by
# default -- without it, a request with no background has nothing to fall back to.
ENABLE_FC = os.environ.get("ENABLE_FC", "1") == "1"
# SD1.5 falls apart above ~768px. Generate small, then let highres_scale upsample.
BASE_SIDE = int(os.environ.get("BASE_SIDE", "640"))

DEVICE = torch.device("cuda")

# ---------------------------------------------------------------------------
# Load at MODULE IMPORT, never inside the handler: FlashBoot snapshots the idle
# process, so anything built here is free on every subsequent cold start.
# ---------------------------------------------------------------------------

_t0 = time.time()

from diffusers import (  # noqa: E402
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.models.attention_processor import AttnProcessor2_0  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from safetensors.torch import load_file as load_safetensors  # noqa: E402
from transformers import CLIPTextModel, CLIPTokenizer  # noqa: E402

TOKENIZER = CLIPTokenizer.from_pretrained(BASE_MODEL, subfolder="tokenizer")
TEXT_ENCODER = CLIPTextModel.from_pretrained(BASE_MODEL, subfolder="text_encoder")
VAE = AutoencoderKL.from_pretrained(BASE_MODEL, subfolder="vae")

TEXT_ENCODER = TEXT_ENCODER.to(device=DEVICE, dtype=torch.float16)
# bfloat16 for the VAE is upstream's choice and it matters: fp16 VAE encode of a flat
# 127-grey field drifts enough to show up as a colour cast across the whole composite.
VAE = VAE.to(device=DEVICE, dtype=torch.bfloat16)
VAE.set_attn_processor(AttnProcessor2_0())


def _build_iclight_unet(in_channels: int, checkpoint: str) -> UNet2DConditionModel:
    """Widen a stock SD1.5 UNet's conv_in and add the IC-Light offset into every tensor."""
    unet = UNet2DConditionModel.from_pretrained(BASE_MODEL, subfolder="unet")

    with torch.no_grad():
        # Zero the new channels, copy the pretrained 4 across, share the original bias.
        # The offset file supplies the real values for channels 4: -- it was trained
        # against exactly this zero-padded layout.
        new_conv_in = torch.nn.Conv2d(
            in_channels,
            unet.conv_in.out_channels,
            unet.conv_in.kernel_size,
            unet.conv_in.stride,
            unet.conv_in.padding,
        )
        new_conv_in.weight.zero_()
        new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
        new_conv_in.bias = unet.conv_in.bias
        unet.conv_in = new_conv_in

    # hf_hub_download rather than torch.hub: it lands in HF_HOME, so Runpod's model
    # caching and the container's own layer cache both see it, and it resumes.
    offset_path = hf_hub_download(repo_id=IC_LIGHT_REPO, filename=checkpoint)
    offset = load_safetensors(offset_path)
    origin = unet.state_dict()
    # strict=True is the point: if a key is missing the architectures disagree and we want
    # to hear about it at boot, not as garbled pixels three weeks later.
    unet.load_state_dict({k: origin[k] + offset[k] for k in origin}, strict=True)
    del offset, origin

    unet = unet.to(device=DEVICE, dtype=torch.float16)
    unet.set_attn_processor(AttnProcessor2_0())

    # Concatenate the conditioning latents onto the noisy latent on the way in. Bound
    # before we shadow the attribute, so `original` is the real nn.Module.forward.
    original = unet.forward

    def hooked_forward(sample, timestep, encoder_hidden_states, **kwargs):
        c_concat = kwargs["cross_attention_kwargs"]["concat_conds"].to(sample)
        # CFG runs cond and uncond as one batch of 2; the conditioning is shared.
        c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
        new_sample = torch.cat([sample, c_concat], dim=1)
        # Emptied before forwarding -- diffusers would otherwise hand our smuggled tensor
        # to the attention processors as if it were a LoRA scale.
        kwargs["cross_attention_kwargs"] = {}
        return original(new_sample, timestep, encoder_hidden_states, **kwargs)

    unet.forward = hooked_forward
    return unet


def _build_pipes(unet):
    """t2i and i2i sharing one UNet. Both are needed: i2i drives the highres refine pass."""
    scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        algorithm_type="sde-dpmsolver++",
        use_karras_sigmas=True,
        steps_offset=1,
    )
    common = dict(
        vae=VAE,
        text_encoder=TEXT_ENCODER,
        tokenizer=TOKENIZER,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        requires_safety_checker=False,
        feature_extractor=None,
        image_encoder=None,
    )
    return StableDiffusionPipeline(**common), StableDiffusionImg2ImgPipeline(**common)


UNET_FBC = _build_iclight_unet(12, "iclight_sd15_fbc.safetensors")
T2I_FBC, I2I_FBC = _build_pipes(UNET_FBC)

UNET_FC = T2I_FC = I2I_FC = None
if ENABLE_FC:
    UNET_FC = _build_iclight_unet(8, "iclight_sd15_fc.safetensors")
    T2I_FC, I2I_FC = _build_pipes(UNET_FC)

print(
    f"[boot] IC-Light on {BASE_MODEL} ready in {time.time() - _t0:.1f}s "
    f"(fbc{' + fc' if ENABLE_FC else ''})",
    flush=True,
)

DEFAULT_ADDED = "best quality, professional product photograph, sharp focus"
DEFAULT_NEGATIVE = (
    "lowres, worst quality, blurry, jpeg artifacts, deformed, distorted product, "
    "warped text, extra objects, watermark"
)

# ---------------------------------------------------------------------------
# Tensor <-> numpy, lifted verbatim from IC-Light so the 127 convention holds.
# ---------------------------------------------------------------------------


def _numpy2pytorch(imgs):
    # /127.0, not /127.5 -- see the module docstring. 127 must land on exactly 0.0.
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0
    return h.movedim(-1, 1)


def _pytorch2numpy(imgs):
    out = []
    for x in imgs:
        y = x.movedim(0, -1) * 127.5 + 127.5
        out.append(y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8))
    return out


def _encode_prompt_inner(txt: str):
    """CLIP is capped at 77 tokens; chunk past that instead of truncating the prompt."""
    max_length = TOKENIZER.model_max_length
    chunk_length = max_length - 2
    id_start, id_end = TOKENIZER.bos_token_id, TOKENIZER.eos_token_id

    def pad(x, p, i):
        return x[:i] if len(x) >= i else x + [p] * (i - len(x))

    tokens = TOKENIZER(txt, truncation=False, add_special_tokens=False)["input_ids"]
    chunks = [[id_start] + tokens[i:i + chunk_length] + [id_end] for i in range(0, max(len(tokens), 1), chunk_length)]
    chunks = [pad(ck, id_end, max_length) for ck in chunks]
    token_ids = torch.tensor(chunks).to(device=DEVICE, dtype=torch.int64)
    return TEXT_ENCODER(token_ids).last_hidden_state


def _encode_prompt_pair(positive: str, negative: str):
    """diffusers requires cond and uncond to be the same shape; repeat the shorter one."""
    c, uc = _encode_prompt_inner(positive), _encode_prompt_inner(negative)
    max_chunk = max(len(c), len(uc))
    c = torch.cat([c] * int(math.ceil(max_chunk / len(c))), dim=0)[:max_chunk]
    uc = torch.cat([uc] * int(math.ceil(max_chunk / len(uc))), dim=0)[:max_chunk]
    c = torch.cat([p[None, ...] for p in c], dim=1)
    uc = torch.cat([p[None, ...] for p in uc], dim=1)
    return c, uc


# ---------------------------------------------------------------------------
# Image framing
# ---------------------------------------------------------------------------


def _snap64(v, lo=256, hi=1024):
    """UNet downsamples 8x on latents that are already 8x down, so sides must be /64."""
    return int(min(hi, max(lo, round(float(v) / 64.0) * 64)))


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _load_image(src, mode="RGB"):
    if not src:
        return None
    if src.startswith("http"):
        with urllib.request.urlopen(src, timeout=60) as r:
            data = r.read()
    else:
        raw = src.split(",", 1)[1] if src.startswith("data:") else src
        data = base64.b64decode(raw)
    return Image.open(io.BytesIO(data)).convert(mode)


def _fit_on_grey(product, width, height, scale, offset_y):
    """Contain-fit an RGBA cut-out onto 127-grey. Returns (uint8 HxWx3, float HxW alpha).

    Transparent margins are trimmed first -- cut-outs routinely arrive with the product
    filling 20% of a huge canvas, and IC-Light would then relight mostly empty grey.
    """
    bbox = product.getchannel("A").getbbox()
    if bbox:
        product = product.crop(bbox)

    # scale < 1 leaves breathing room at the frame edge. IC-Light's own examples all have
    # it, and it also gives the contact shadow somewhere to fall.
    box_w, box_h = max(1, int(width * scale)), max(1, int(height * scale))
    ratio = min(box_w / product.width, box_h / product.height)
    new_size = (max(1, round(product.width * ratio)), max(1, round(product.height * ratio)))
    product = product.resize(new_size, Image.LANCZOS)

    canvas = Image.new("RGBA", (width, height), (127, 127, 127, 0))
    # Positive offset_y pushes the product down the frame, which is usually what you want:
    # it leaves scene above and floor below, rather than floating the product dead centre.
    top = _clamp((height - new_size[1]) // 2 + int(offset_y * height), 0, max(0, height - new_size[1]))
    canvas.paste(product, ((width - new_size[0]) // 2, top))

    alpha = np.asarray(canvas.getchannel("A"), dtype=np.float32) / 255.0
    rgb = np.asarray(canvas.convert("RGB"), dtype=np.float32)
    # IC-Light's own matting formula. Handles the empty margin AND soft cut-out edges,
    # where the source RGB under a partly-transparent pixel is often junk.
    rgb = 127.0 + (rgb - 127.0) * alpha[..., None]
    return rgb.clip(0, 255).astype(np.uint8), alpha


def _resize_and_center_crop(image, width, height):
    """Cover-fit, for backgrounds -- cropping a scene is fine, squashing it is not."""
    scale = max(width / image.width, height / image.height)
    rw, rh = int(round(image.width * scale)), int(round(image.height * scale))
    resized = image.resize((rw, rh), Image.LANCZOS)
    left, top = (rw - width) / 2, (rh - height) / 2
    return np.asarray(resized.crop((left, top, left + width, top + height)))


def _gradient_bg(direction, width, height):
    """Upstream's fc initial-latent gradients: a crude 'light comes from over there' hint."""
    # Bright end = where the light comes from.
    if direction in ("left", "right"):
        ramp = np.linspace(255, 0, width) if direction == "left" else np.linspace(0, 255, width)
        img = np.tile(ramp, (height, 1))
    elif direction in ("top", "bottom"):
        ramp = np.linspace(255, 0, height) if direction == "top" else np.linspace(0, 255, height)
        img = np.tile(ramp[:, None], (1, width))
    else:
        return None
    return np.stack((img,) * 3, axis=-1).astype(np.uint8)


def _apply_contact_shadow(bg, alpha, strength, squash, offset_x, softness):
    """Darken the background where the product will sit, before it becomes conditioning.

    This is the cheap stand-in for libcom's shadow generator, and it works because fbc
    reproduces the background's luminance structure in its output: give the model a soft
    dark patch under the product and it renders a shadow there rather than leaving the
    product hovering. The silhouette is squashed toward the ground plane and pinned at the
    contact point, which is what a real shadow from a high-ish key light looks like.
    """
    ys, xs = np.where(alpha > 0.05)
    if len(ys) == 0:
        return bg

    top, bottom, left, right = ys.min(), ys.max(), xs.min(), xs.max()
    crop = (alpha[top:bottom + 1, left:right + 1] * 255).astype(np.uint8)
    squashed_h = max(1, int(round(crop.shape[0] * squash)))
    crop_img = Image.fromarray(crop).resize((crop.shape[1], squashed_h), Image.LANCZOS)

    canvas = Image.new("L", (bg.shape[1], bg.shape[0]), 0)
    # bottom edge stays put: that is where the product meets the surface.
    canvas.paste(crop_img, (int(left + offset_x), int(bottom - squashed_h + 1)))
    canvas = canvas.filter(ImageFilter.GaussianBlur(radius=softness))

    shadow = np.asarray(canvas, dtype=np.float32)[..., None] / 255.0
    return (bg.astype(np.float32) * (1.0 - strength * shadow)).clip(0, 255).astype(np.uint8)


def _preserve_detail(result, product_rgb, alpha, amount):
    """Add the original product's high frequencies back inside the mask.

    SD1.5 at 640px cannot hold cedar grain or the etched lettering on a stainless panel;
    it will smooth them into plausible mush. This transplants the source's fine detail
    onto the relit result while leaving the new lighting (all low frequency) alone.
    Off by default: if the relight shifted the product's silhouette at all, the transplant
    ghosts, and on a badly-drifted result that is worse than soft grain.
    """
    radius = max(1.0, result.shape[1] / 256.0)
    lo_res = np.asarray(Image.fromarray(result).filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32)
    lo_src = np.asarray(Image.fromarray(product_rgb).filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32)
    # Frequency separation, not additive sharpening: keep the relit low frequencies (they
    # ARE the new lighting) and swap in the source's high frequencies (they ARE the grain
    # and lettering the model smoothed away). Adding instead would double the contrast.
    transplanted = lo_res + (product_rgb.astype(np.float32) - lo_src)
    # Squared alpha so feathered cut-out edges -- exactly where a misalignment would
    # show -- stay on the relit result rather than the transplant.
    mask = (alpha[..., None] ** 2) * amount
    blended = result.astype(np.float32) * (1.0 - mask) + transplanted * mask
    return blended.clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _encode_cond(fg, bg):
    """VAE-encode the conditioning images into the channels bolted onto conv_in."""
    images = [fg] if bg is None else [fg, bg]
    x = _numpy2pytorch(images).to(device=VAE.device, dtype=VAE.dtype)
    # .mode(), not .sample(): conditioning must be deterministic or the seed lies.
    latents = VAE.encode(x).latent_dist.mode() * VAE.config.scaling_factor
    if bg is None:
        return latents
    # fbc wants fg and bg stacked on the CHANNEL axis, not the batch axis.
    return torch.cat([c[None, ...] for c in latents], dim=1)


# The diffusers pipelines guard their own denoise loops, but the VAE encode/decode and
# CLIP calls below are ours and would otherwise build an autograd graph nobody reads --
# pure waste, and enough extra VRAM to OOM a highres pass on a 16GB card.
@torch.inference_mode()
def handler(job):
    job_input = job.get("input") or {}
    started = time.time()
    try:
        product = _load_image(
            job_input.get("image_url") or job_input.get("image_base64") or job_input.get("image"),
            mode="RGBA",
        )
        if product is None:
            return {"error": "image_url or image_base64 is required"}

        background = _load_image(
            job_input.get("background_url")
            or job_input.get("background_base64")
            or job_input.get("background"),
            mode="RGB",
        )

        prompt = (job_input.get("prompt") or "").strip()
        if not prompt and background is None:
            return {"error": "prompt is required when no background is supplied"}

        # A background means we can relight INTO the scene (fbc). Without one there is
        # nothing to condition on, so fall back to relighting from the prompt alone (fc).
        mode = (job_input.get("mode") or ("fbc" if background is not None else "fc")).lower()
        if mode not in ("fbc", "fc"):
            return {"error": f"unknown mode '{mode}' (expected 'fbc' or 'fc')"}
        if mode == "fbc" and background is None:
            return {"error": "mode 'fbc' needs background_url or background_base64"}
        if mode == "fc" and not ENABLE_FC:
            return {"error": "text-conditioned mode is disabled on this endpoint (ENABLE_FC=0)"}

        t2i, i2i = (T2I_FBC, I2I_FBC) if mode == "fbc" else (T2I_FC, I2I_FC)

        # Size: caller's if given, else the scene's aspect at BASE_SIDE, else upstream's
        # 512x640 portrait. Never the product's aspect -- the output frames the scene.
        if job_input.get("width") and job_input.get("height"):
            width, height = _snap64(job_input["width"]), _snap64(job_input["height"])
        elif background is not None:
            bw, bh = background.size
            width, height = (
                (_snap64(BASE_SIDE), _snap64(BASE_SIDE * bh / bw))
                if bw >= bh
                else (_snap64(BASE_SIDE * bw / bh), _snap64(BASE_SIDE))
            )
        else:
            width, height = 512, 640

        steps = _clamp(int(job_input.get("steps") or (20 if mode == "fbc" else 25)), 1, 100)
        # fbc tolerates normal CFG; fc is trained to run near 2.0 and burns at 7.
        cfg = float(job_input.get("cfg") or (7.0 if mode == "fbc" else 2.0))
        seed = int(job_input.get("seed") if job_input.get("seed") is not None else random.randint(0, 2**31 - 1))
        highres_scale = _clamp(float(job_input.get("highres_scale") or 1.0), 1.0, 3.0)
        # The denoise floors are not taste, they are div-by-zero guards: both values divide
        # `steps` below to work out how long a schedule to build for a partial denoise.
        highres_denoise = _clamp(float(job_input.get("highres_denoise") or 0.5), 0.05, 1.0)
        lowres_denoise = _clamp(float(job_input.get("lowres_denoise") or 0.9), 0.05, 1.0)
        fg_scale = _clamp(float(job_input.get("fg_scale") or 0.9), 0.1, 1.0)
        fg_offset_y = _clamp(float(job_input.get("fg_offset_y") or 0.0), -0.5, 0.5)
        # .get(key, default) not `or`, so an explicit 0 switches the shadow off.
        shadow_strength = _clamp(float(job_input.get("shadow_strength", 0.5)), 0.0, 1.0)
        shadow_squash = _clamp(float(job_input.get("shadow_squash") or 0.22), 0.02, 1.0)
        shadow_offset_x = _clamp(float(job_input.get("shadow_offset_x") or 0.0), -0.5, 0.5)
        preserve = _clamp(float(job_input.get("preserve_detail") or 0.0), 0.0, 1.0)
        light_direction = (job_input.get("light_direction") or "none").lower()

        alpha_channel = np.asarray(product.getchannel("A"))
        has_cutout = bool((alpha_channel < 250).any())
        warning = None
        if not has_cutout:
            warning = (
                "product image has no transparency, so its original background is being fed "
                "in as part of the subject. Cut it out first with the BiRefNet worker and "
                "pass the RGBA result."
            )

        def prepare(w, h):
            """Build the conditioning pair at a given size. Called again for highres."""
            fg, alpha = _fit_on_grey(product, w, h, fg_scale, fg_offset_y)
            if mode == "fbc":
                bg = _resize_and_center_crop(background, w, h)
                if shadow_strength > 0 and has_cutout:
                    bg = _apply_contact_shadow(
                        bg, alpha, shadow_strength, shadow_squash,
                        shadow_offset_x * w,
                        float(job_input.get("shadow_softness") or max(2.0, w * 0.02)),
                    )
            else:
                bg = None
            return fg, bg, alpha

        rng = torch.Generator(device=DEVICE).manual_seed(seed)
        fg, bg, alpha = prepare(width, height)
        conds, unconds = _encode_prompt_pair(
            positive=", ".join(p for p in [prompt, job_input.get("added_prompt", DEFAULT_ADDED)] if p),
            negative=job_input.get("negative_prompt") or DEFAULT_NEGATIVE,
        )

        # width/height are read by the t2i pipeline and silently swallowed by the i2i one
        # (it has **kwargs and takes its size from the input latents instead). Harmless,
        # and kept on both for symmetry -- do not "fix" it by removing them from t2i.
        shared = dict(
            prompt_embeds=conds,
            negative_prompt_embeds=unconds,
            width=width,
            height=height,
            num_images_per_prompt=1,
            generator=rng,
            output_type="latent",
            guidance_scale=cfg,
            cross_attention_kwargs={"concat_conds": _encode_cond(fg, bg)},
        )

        init_bg = _gradient_bg(light_direction, width, height) if mode == "fc" else None
        if init_bg is None:
            latents = t2i(num_inference_steps=steps, **shared).images
        else:
            # fc has no background channel, so a light direction can only be expressed by
            # seeding the initial latent with a gradient and denoising most of it away.
            init = _numpy2pytorch([init_bg]).to(device=VAE.device, dtype=VAE.dtype)
            init = VAE.encode(init).latent_dist.mode() * VAE.config.scaling_factor
            latents = i2i(
                image=init,
                strength=lowres_denoise,
                num_inference_steps=int(round(steps / lowres_denoise)),
                **shared,
            ).images
        latents = latents.to(VAE.dtype) / VAE.config.scaling_factor

        if highres_scale > 1.0:
            # Upscale the decoded pixels, re-encode, and refine at the larger size with
            # conditioning rebuilt to match. A plain upscale would just be a blurry crop.
            pixels = _pytorch2numpy(VAE.decode(latents).sample)
            hw, hh = _snap64(width * highres_scale, hi=2048), _snap64(height * highres_scale, hi=2048)
            pixels = [np.asarray(Image.fromarray(p).resize((hw, hh), Image.LANCZOS)) for p in pixels]
            latents = _numpy2pytorch(pixels).to(device=VAE.device, dtype=VAE.dtype)
            latents = VAE.encode(latents).latent_dist.mode() * VAE.config.scaling_factor

            fg, bg, alpha = prepare(hw, hh)
            latents = i2i(
                image=latents.to(dtype=torch.float16),
                strength=highres_denoise,
                num_inference_steps=int(round(steps / highres_denoise)),
                **{**shared, "width": hw, "height": hh,
                   "cross_attention_kwargs": {"concat_conds": _encode_cond(fg, bg)}},
            ).images
            latents = latents.to(VAE.dtype) / VAE.config.scaling_factor
            width, height = hw, hh

        result = _pytorch2numpy(VAE.decode(latents).sample)[0]

        if preserve > 0 and has_cutout:
            result = _preserve_detail(result, fg, alpha, preserve)

        buf = io.BytesIO()
        Image.fromarray(result).save(buf, format="PNG")
        out = {
            "image_url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
            "width": width,
            "height": height,
            "mode": mode,
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "model": f"{IC_LIGHT_REPO}/iclight_sd15_{mode}",
            "base_model": BASE_MODEL,
            "generation_seconds": round(time.time() - started, 2),
        }
        if warning:
            out["warning"] = warning

        # Debug aid: what the model was actually shown, after framing and shadow injection.
        if job_input.get("return_conditioning"):
            for name, arr in (("conditioning_fg_url", fg), ("conditioning_bg_url", bg)):
                if arr is None:
                    continue
                cbuf = io.BytesIO()
                Image.fromarray(arr).save(cbuf, format="PNG")
                out[name] = "data:image/png;base64," + base64.b64encode(cbuf.getvalue()).decode()
        return out
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


import runpod  # noqa: E402

runpod.serverless.start({"handler": handler})
