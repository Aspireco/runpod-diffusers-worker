# Wan 2.2 S2V — speech-driven video

Give it a **voice track** and a **reference image**; it returns a lip-synced performance
with that voice muxed into the file.

## The thing to understand first

**S2V does not generate a voice. It consumes one.** The name is *Speech*-to-Video. If you
want a spokesperson reading a script, the voice comes from somewhere else and S2V animates
to it:

```
script  ->  voice-qwen (Qwen3-TTS, apache-2.0)  ->  voice.wav
                                                        |
reference image  ------------------------------------>  S2V  ->  talking video (with audio)
```

Models that invent video *and* audio together from a text prompt are a different category —
MiniMax H3 and LTX-2.5. MiniMax's §V.4 forbids displaying its **outputs** outside its
Applicable Territory, which excludes the United States; see `stack-docs/OPEN-QUESTIONS.md`.

## Why ComfyUI and not diffusers

Two independent blockers, both checked rather than assumed:

1. **`WanSpeechToVideoPipeline` does not exist in any diffusers release.** Verified against
   the `__init__.py` of v0.37.1, v0.38.0, v0.39.0, v0.40.0 and `main`. diffusers ships
   `WanPipeline`, `WanImageToVideoPipeline`, `WanVACEPipeline`, `WanAnimatePipeline` — no
   speech variant.
2. **The upstream repo isn't in diffusers layout.** `Wan-AI/Wan2.2-S2V-14B` ships the
   original Wan format — raw `diffusion_pytorch_model-*.safetensors` shards,
   `Wan2.1_VAE.pth`, `models_t5_umt5-xxl-enc-bf16.pth` — and **no `model_index.json`**, so
   `DiffusionPipeline.from_pretrained` cannot read it either.

ComfyUI has native nodes: `WanSoundImageToVideo`, `AudioEncoderLoader`, `AudioEncoderEncode`
(and `WanSoundImageToVideoExtend` for longer takes). All three were confirmed present in a
live 0.26.2 registry before the build assertion was written.

## What's baked in

| file | size | role |
| --- | --- | --- |
| `wan2.2_s2v_14B_bf16.safetensors` | 32.6 GB | the transformer |
| `wav2vec2_large_english_fp16.safetensors` | 0.63 GB | encodes the voice into conditioning |
| `umt5_xxl_fp16.safetensors` | 11.4 GB | text encoder (`CLIPLoader` type `wan`) |
| `wan_2.1_vae.safetensors` | 0.25 GB | **2.1**, not 2.2 |

All **apache-2.0**, from Comfy-Org's repack of Wan's own weights. Wan is the only top-tier
video family with no territory clause, no revenue cap and no attribution requirement.

An `fp8_scaled` build of the transformer exists at half the size if a 48 GB card is ever the
target; bf16 is baked here because this lane runs on an 80 GB card.

## Traps

- **The VAE is `wan_2.1_vae`, not `wan2.2_vae`.** Both ship in the same repo and the names
  differ by one character. Picking wrong does not raise — it degrades the output.
- **Feed the audio to `CreateVideo` as well as to the encoder.** The workflow wires
  `LoadAudio` into both. Omit the second and you get a silent file whose lip-sync has
  nothing to sync against on playback — the single easiest thing to get wrong here.
- **`length: 77` is the node's own default and matches the audio window it expects.**
  Raising it without lengthening the audio gives you silence at the tail.
- **`init_extra_nodes()` is `async def` in current ComfyUI.** Calling it bare returns a
  coroutine, registers nothing and raises nothing — so a node assertion written that way
  silently tests an empty registry. The build awaits it.
- **`comfy model download` has returned success on a truncated file.** The build checks real
  file sizes against floors rather than trusting the exit code.

## Workflows

`workflows/wan22_s2v_voice.json` — voice + reference image -> talking video.
`workflows/wan22_i2v_80g_two_expert.json` — the separate 80 GB i2v config: high-noise expert
takes steps 0-20 and passes the latent on **with leftover noise**, low-noise finishes 20-40.
Running either expert alone is half a model.

Both are API-format graphs built from signatures read off a live registry, with every node
reference checked.
