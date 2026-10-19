# audio-clean — Voice Isolator equivalent (T4, 16 GB)

Pulls a voice out of noise/music, removes the voice to keep only the background, or just denoises.
Demucs v4 (MIT, facebookresearch/demucs) does the source separation via its own official
`demucs.api.Separator`; DeepFilterNet 3 (MIT OR Apache-2.0, Rikorose/DeepFilterNet) does the speech
denoising via its own official `df.enhance.init_df`/`enhance`. Both are called directly — the only code
in `handler.py` is the thin per-job glue (fetch/decode, pick stems, resample back, mux).

## Premade search

| Looked for | Found | Used / rejected |
|---|---|---|
| A speech/music source-separation model with an official high-level Python API, so stems are not hand-rolled | `demucs.api.Separator` (facebookresearch/demucs) — `Separator(model, repo, device)` + `.separate_audio_file(path)` returns a dict of stems | **Used.** This is the exact `--repo` local-directory mechanism the official CLI itself supports for fully offline use (`demucs/pretrained.py`'s `add_model_flags`), not a workaround. |
| A speech denoiser with a premade checkpoint and an official call path | `df.enhance.init_df()` / `enhance()` (Rikorose/DeepFilterNet, DeepFilterNet3) | **Used.** `handler.py`'s `_denoise()` mirrors `df/enhance.py`'s own `main()` per-file loop (load at the model's rate → enhance → resample back → save) almost line for line. |
| An alternative offline Demucs loading path (e.g. re-downloading at first boot instead of baking) | `RemoteRepo` (downloads via `torch.hub.load_state_dict_from_url` from `dl.fbaipublicfiles.com`) | **Rejected.** No fetch-job pipeline for this lane (task's own ownership boundary); `LocalRepo` + baked weights removes the runtime network dependency and matches the task's "small weights baked in" guidance. |
| A way to get DeepFilterNet3 running on this base image's Python 3.12 without compiling Rust | A cp312 wheel for `deepfilterlib` on PyPI | **Not found.** Checked every release up to and including the current latest (0.5.6, PyPI JSON, 2026-10-17): wheels stop at cp311. Repo's last release was 2024-10-17. **Rejected** a second Python 3.11 venv just to get a prebuilt wheel (more moving parts, cross-venv IPC in the handler) in favour of building from the sdist via `cargo`/`rustc` — `deepfilterlib`'s own `Cargo.toml` pins `edition = "2021"` and `pyo3 = "0.19"`, both comfortably within what Ubuntu 24.04's packaged `cargo` (1.75.0) supports. |
| Whether `demucs==4.1.0` needs anything beyond its own declared (non-extra) dependencies | `demucs/pretrained.py` has an **unconditional** `from dora.log import fatal, bold` at module level, but PyPI's requires_dist gates `dora-search` behind demucs's own `"train"` extra | **Added `dora-search` explicitly.** A plain `pip install demucs` would otherwise leave this import unmet. Checked `dora-search`'s own dependency chain for licence issues (next section) since it was not in the task's original model list. |

## Licence table

| Component | Source | Licence | Verified |
|---|---|---|---|
| Demucs v4 (code + htdemucs/htdemucs_ft weights) | `github.com/facebookresearch/demucs`, weights at `dl.fbaipublicfiles.com/demucs/...` | MIT | `LICENSE` fetched directly (Meta Platforms, Inc.) + PyPI classifier `License :: OSI Approved :: MIT License` for the 4.1.0 release |
| DeepFilterNet 3 (code + checkpoint) | `github.com/Rikorose/DeepFilterNet`, checkpoint at `github.com/Rikorose/DeepFilterNet/raw/main/models/DeepFilterNet3.zip` | MIT OR Apache-2.0 (dual) | `LICENSE-MIT` and `LICENSE-APACHE` both fetched directly (Hendrik Schröter, 2021) |
| deepfilterlib (native DF core) | PyPI `deepfilterlib` | MIT | same repo/licence as DeepFilterNet above (`pyDF/` is part of the same dual-licensed tree) |
| dora-search | PyPI `dora-search` | MIT | PyPI classifier `License :: OSI Approved :: MIT License` |
| ↳ omegaconf | PyPI `omegaconf` | BSD | PyPI classifier `License :: OSI Approved :: BSD License` |
| ↳ retrying | PyPI `retrying` (no PyPI classifier) | Apache-2.0 | `LICENSE` fetched directly from `github.com/rholder/retrying` |
| ↳ submitit | PyPI `submitit` | MIT | PyPI classifier `License :: OSI Approved :: MIT License` |
| ↳ treetable | PyPI `treetable` | Unlicense (public domain) | PyPI `license` field: "Unlicense license" |
| einops, huggingface-hub, julius, lameenc, pyyaml, safetensors, sphn, tqdm, appdirs, loguru, numpy, packaging, requests, sympy | PyPI (demucs/deepfilternet's own direct deps) | well-established permissive licences | not individually re-verified — all long-standing, widely used ML-stack packages |

No GPL/AGPL/non-commercial terms anywhere in this stack. Nothing triggered the task's "STOP and report"
clause.

## Handler contract

```json
// input
{
  "audio": "https://.../noisy.wav",      // OR "video": "...", exactly one required
  "mode": "isolate_voice",               // optional: "isolate_voice" (default) | "remove_voice" | "denoise"
  "max_seconds": 900                     // optional, default shown; refuses longer audio with an error
}
```

```json
// output
{
  "audio_url": "data:audio/wav;base64,...",
  "mode": "isolate_voice",
  "duration_seconds": 12.34,
  "sample_rate": 44100,
  "generation_seconds": 3.9,
  "model": "htdemucs",                   // only present for isolate_voice / remove_voice
  "video_url": "data:video/mp4;base64,..." // only present when the input was a video
}
```

On failure: `{"error": "<Type>: <message>"}` — the handler never raises.

**Output sample rate** always matches the *input's own* sample rate (both Demucs- and
DeepFilterNet-involving paths resample back before writing), so the result is a drop-in replacement for
the source track regardless of which mode ran.

**`isolate_voice`** downmixes the Demucs "vocals" stem to mono before DeepFilterNet (DeepFilterNet3 is a
mono speech model, and a single cleaned voice track is the expected product of a cleanup tool regardless
of the source's channel count). **`remove_voice`** stays at Demucs's own native channel count/rate
(stereo, 44100 Hz) with no DeepFilterNet step — the spec is "stems summed", not also denoised.
**`denoise`** also downmixes to mono for the same reason as `isolate_voice`.

**`DEMUCS_MODEL`** (env var, default `htdemucs`) is not a job-input field — the handler contract above is
implemented exactly as given. Both `htdemucs` (single model, one forward pass, the upstream CLI's own
default) and `htdemucs_ft` (bag of 4 fine-tuned sources, ~4x the compute, typically slightly better
separation) are baked into the image; redeploying with `DEMUCS_MODEL=htdemucs_ft` switches the whole lane
to the slower/higher-quality variant without a rebuild.

## torch.load / weights_only note (flagged, not hidden)

`demucs/states.py`'s `load_model()` calls `torch.load(path, 'cpu')` with no `weights_only` override, and
every `.th` checkpoint it loads pickles an actual class reference (e.g. `HDemucs`) plus `args`/`kwargs`/
`state` — not weights-only tensors. torch ≥ 2.6 (this image ships 2.8) defaults `weights_only=True`,
which would reject that load. `handler.py` and `build_check.py` both carry a narrow monkeypatch that
forces `weights_only=False` for exactly the two boot-time loads (Demucs + DeepFilterNet3), then restore
torch's own secure default immediately after — every checkpoint involved was downloaded and size-checked
by this worker's own Dockerfile, never supplied by a caller at runtime, so this reverts to the pre-2.6
default only for trusted, build-time-baked files, not for anything request-time. This is reasoned from
reading `demucs/states.py` directly (2026-10-17), not from a real job — the task's ownership boundary
rules out starting one — but `build_check.py` does perform a real construct-and-load of both Demucs
variants and the DeepFilterNet3 checkpoint at ACR build time, so the ACR build itself is the real proof.

## VRAM and speed on a T4 16GB

**Not measured** (the task's hard boundary is no GPU jobs). Reasoned estimates only:

- **Demucs htdemucs**: ~80M-parameter hybrid transformer, one forward pass per chunk (default `segment`,
  `shifts=1`, `split=True` chunking for memory safety). Demucs' own published benchmarks put a few
  minutes of stereo audio at well under real-time on a single modern GPU; a T4 is slower than the cards
  those benchmarks typically use, so plan for roughly real-time to a few-times-real-time, not instant.
  `htdemucs_ft` (bag of 4) costs ~4x that.
- **DeepFilterNet3**: designed to run in real time even on CPU (its whole premise is a low-complexity
  recurrent architecture for embedded/real-time use), so on a T4 GPU it should be a small fraction of the
  Demucs cost in any mode that uses both.
- **VRAM**: both models' weights are well under 500MB combined; resident VRAM for activations at these
  model sizes should be a small fraction of the T4's 16GB even for the full `max_seconds=900` ceiling,
  but this is reasoned from model size, not measured.
- **Cold start**: baked weights mean zero download time on cold start — load time is just two small
  `torch.load` calls plus Rust-extension/import overhead, expected to be single-digit seconds, not the
  minutes seen on lanes that stage tens of gigabytes from the Azure Files share.

**Measure this for real with a real job** before relying on these numbers — this task's boundary does not
include running one.

## Files

| file | what it is |
|---|---|
| `Dockerfile` | the image: bakes both Demucs variants + DeepFilterNet3, builds `deepfilterlib` from source via `cargo` (no cp312 wheel exists upstream), runs `build_check.py` |
| `handler.py` | the serverless worker: boots both models once, then per job fetches/decodes, runs the chosen mode's pipeline, resamples back, muxes video if requested |
| `build_check.py` | build-time import **and load** check — constructs both Demucs variants and runs `init_df()` against the baked DeepFilterNet3 checkpoint, so a bad download or the weights_only mismatch fails the build, not a billed GPU |
| `README.md` | this file |
