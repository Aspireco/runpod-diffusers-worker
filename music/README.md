# music — ACE-Step 1.5 (T4, 16GB)

Text-to-music for ad beds and background tracks. ACE-Step 1.5 (MIT, `github.com/ace-step/ACE-Step-1.5`)
replaces the earlier v1 (`ACE-Step/ACE-Step-v1-3.5B`) integration on this same lane. `handler.py` calls
the project's own `AceStepHandler` + `generate_music` (`acestep.inference`) directly — the only code
here is per-job glue (fetch an optional reference clip, build `GenerationParams`/`GenerationConfig`,
save, base64-encode).

## Premade search

| Looked for | Found | Used / rejected |
|---|---|---|
| An MIT or Apache-licensed music-generation model with no revenue-based licence termination | Stable Audio (Stability Community Licence — terminates above $1M revenue aggregated across affiliates); ACE-Step 1.5 (MIT, no revenue condition) | **Used ACE-Step 1.5.** Carried over from this lane's own existing choice (the pre-existing v1 Dockerfile/handler.py this task replaces); re-verified at source for the 1.5 release specifically rather than assumed. |
| ACE-Step 1.5's own supported low-level Python entry point, so this handler calls official code rather than re-implementing the pipeline | `acestep.handler.AceStepHandler` + `acestep.inference.generate_music` (documented in the project's own `docs/en/INFERENCE.md` "Quick Start") | **Used.** Same two objects their own Quick Start example constructs; `handler.py` is the thin RunPod-contract glue around them. |
| A low-VRAM path that avoids the optional 5Hz language-model ("thinking") stage entirely | `docs/en/GPU_COMPATIBILITY.md` → "Memory Optimization Tips: Very Low VRAM (≤6GB): Use DiT-only mode without LM initialization"; confirmed in code by reading `generate_music.py` directly: `use_lm = (params.thinking or need_lm_for_cot) and llm_handler is not None and llm_handler.llm_initialized and not skip_lm` | **Used — DiT-only, `llm_handler=None`.** This lane's contract already supplies tags/lyrics directly; there is nothing for the LM's chain-of-thought stage to add. Also avoids `nano-vllm`, which is not on PyPI at all (see Dockerfile) and would otherwise be a hard, unresolvable pip dependency of the project's own `pyproject.toml`. |
| Whether the Qwen3-Embedding-0.6B component (bundled in the main model repo) belongs to the optional LM or to the DiT itself | Read `init_service_loader_components.py` directly: `_load_text_encoder_and_tokenizer` always loads `<checkpoint_dir>/Qwen3-Embedding-0.6B` and assigns `self.text_encoder`/`self.text_tokenizer`; `conditioning_text.py`'s `_prepare_text_conditioning_inputs` tokenizes `caption`/`lyrics` with exactly those two objects | **Required regardless of LM use.** Kept in the weights set; only `acestep-5Hz-lm-1.7B` (the actual optional LM) is dead weight on disk for this lane (see "Weights" below). |

## Licence table

| Component | Source | Licence | Verified |
|---|---|---|---|
| ACE-Step 1.5 (code) | `github.com/ace-step/ACE-Step-1.5` | MIT | `LICENSE` fetched directly from the repo (`Copyright (c) 2026 ACEStep`), and the project's own README "License & Disclaimer" section: "This project is licensed under [MIT](./LICENSE)" |
| ACE-Step/Ace-Step1.5 (weights: turbo DiT, VAE, Qwen3-Embedding-0.6B, acestep-5Hz-lm-1.7B) | `huggingface.co/ACE-Step/Ace-Step1.5` | MIT | HF API `model_info` `cardData.license` = `"mit"` and tag `license:mit`, 2026-10-17 |

No GPL/AGPL/non-commercial terms anywhere in this stack. Nothing triggered the task's "STOP and
report" clause — ACE-Step 1.5 is MIT, same as the project it replaces claimed to be.

**Caveat carried over from the old v1 lane, not resolved by the 1.5 upgrade:** the "licensed /
royalty-free training data" claim appears on marketing pages but is not documented in the technical
report. For paid advertising that is a judgement call, not a settled fact.

## Weights

Several GB — far past this repo's ~500MB bake budget, so this lane fetches from the Azure Files
share like `tts`/`voice-qwen3`/`voice-design`, not baked like `audio-clean`.

`gpu/manifests/music.tsv`: every file in `ACE-Step/Ace-Step1.5` except `.gitattributes` and
`README.md` (sizes from `HfApi().model_info(repo, files_metadata=True)`, 2026-10-17) — **26 files,
10,092,095,357 bytes = 10.09 GB**:

| Folder | What | Bytes | Used at inference? |
|---|---|---:|---|
| `acestep-v15-turbo/` | DiT (2B) + its own `trust_remote_code=True` model/config code + silence latent | 4,791,777,953 | **Yes — the model this lane serves** |
| `vae/` | `AutoencoderOobleck` (Stable-Audio-style VAE) | 337,431,813 | **Yes** |
| `Qwen3-Embedding-0.6B/` | The DiT's own text encoder + tokenizer (see "Premade search") | 1,207,471,006 | **Yes** |
| `acestep-5Hz-lm-1.7B/` | The optional 5Hz "thinking" language model | 3,755,412,617 | **No — never loaded.** Staged only because `check_main_model_exists()` (`acestep/model_downloader.py`) requires every file in `MAIN_MODEL_COMPONENTS` (all four folders above) to already contain weights before `initialize_service()` will skip a live HuggingFace download attempt. Its ~3.7GB is the price of a reliable no-network cold start, not a VRAM cost: this handler never constructs an `LLMHandler` or imports `acestep.llm_inference`. |
| `config.json` | Root HF `AutoModel`/`AutoConfig` pointer (for the `pipeline("text-to-audio", ..., trust_remote_code=True)` shortcut shown on the HF model card) | 1,968 | Not on this lane's own load path (`initialize_service` resolves `<checkpoint_dir>/acestep-v15-turbo` directly) — kept anyway; negligible size |

`weights.repo` = `ACE-Step/Ace-Step1.5`, `weights.dest` = `/runpod-volume/acestep-v15/checkpoints`
(= `ACESTEP_CHECKPOINTS_DIR`), `weights.manifest` = `/manifests/music.tsv` — see the proposed
`lanes.json` entry in the task report.

## Handler contract

```json
// input
{
  "tags": "warm acoustic, upbeat, corporate",   // or "prompt" / "caption" -- required
  "lyrics": "[Instrumental]",                    // optional; "", "[inst]", "[Instrumental]" all mean instrumental
  "duration": 30,                                // seconds, 10-600; default 30
  "infer_step": 8,                               // or "num_inference_steps"; default 8 (turbo's own recommendation)
  "guidance_scale": 7.0,                         // accepted but ignored by turbo (it forces 1.0 itself)
  "seed": -1,                                    // -1 = random
  "language": "en",                              // or "vocal_language"; default "unknown" (no LM auto-detect)
  "reference_audio": "https://... or data:audio/..."  // optional style reference, data: or https URL
}
```

```json
// output
{
  "audio_url": "data:audio/wav;base64,...",
  "duration": 30.02,
  "sample_rate": 48000,
  "tags": "warm acoustic, upbeat, corporate",
  "instrumental": true,
  "model": "acestep-v15-turbo",
  "seed": 1234567,
  "generation_seconds": 4.1
}
```

On failure: `{"error": "<Type>: <message>"}` — the handler never raises.

**Not implemented** (future extension, out of this task's scope): `task_type` switching to
`cover`/`repaint`/`extract`/`complete`/`lego` (ACE-Step 1.5's other direct-conditioning tasks, which
also skip the LM) and `src_audio`. Only plain `text2music` (optionally style-guided by
`reference_audio`) is wired up.

## VRAM and speed on a T4 16GB

**Not measured** (the task's hard boundary is no GPU jobs). Reasoned estimate:

- Resident weights with no CPU offload: DiT 2B (~4.79GB) + VAE (~0.34GB) + Qwen3-Embedding-0.6B
  (~1.21GB) ≈ **6.3GB**, comfortably inside a T4's 16GB with headroom for activations even at the
  full 600s duration ceiling. The LM (3.75GB on disk) is never loaded (see "Weights" above), so it
  costs nothing here.
- T4 is Turing (compute capability 7.5), which has no native bf16 tensor-core path; ACE-Step 1.5's
  own `initialize_service()` detects this (`gpu_config.cuda_supports_bfloat16()`) and falls back to
  float16 automatically — same 2 bytes/param, no size change, just confirmed so the VRAM arithmetic
  above isn't silently wrong on this card.
- Official project numbers (`README.md` "Abstract"): "under 2 seconds per full song on an A100 and
  under 10 seconds on an RTX 3090" for the 2B turbo DiT at 8 steps — a T4 is slower than a 3090, so
  plan for several-times-real-time generation per clip, not sub-10-second, but still fast relative to
  a typical ad-bed turnaround.

**Measure this for real with a real job** before relying on these numbers.

## Files

| file | what it is |
|---|---|
| `Dockerfile` | the image: no weights baked in, `--no-deps` install of ACE-Step 1.5 plus a hand-picked minimal dependency set, runs `build_check.py` |
| `handler.py` | the serverless worker: loads `AceStepHandler` once at import, then per job builds `GenerationParams`/`GenerationConfig` and calls `generate_music` |
| `build_check.py` | build-time import **and construct** check (no GPU, no weights at ACR build time — see its own docstring for why it stops where it does) |
| `README.md` | this file |
