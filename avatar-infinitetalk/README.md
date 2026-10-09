# avatar-infinitetalk worker

InfiniteTalk (MeiGen-AI, Apache-2.0 weights) on ComfyUI + the WanVideoWrapper nodes, run as
a RunPod serverless handler. Backs the Studio lane `video/avatar/infinitetalk`
(endpoint `5v35kjtm7eknd2`, "mos-infinitetalk").

## Provenance

Vendored 2026-10-09 from `wlsdml1114/Infinitetalk_Runpod_hub` (RunPod Hub template,
no LICENSE file in that repo — code is all-rights-reserved by default; only the model
weights it downloads at build time are Apache-2.0). We don't have redistribution rights
from the author; this copy exists only to patch a crash and keep our own pinned, buildable
record of the exact image this lane runs, the way every other lane in this repo is built.
**Flag to John before treating this as settled**: either get the author's OK / a licence
grant, or write our own handler against the same public ComfyUI graph and drop this file.

## What was patched (2026-10-09)

`handler.py` had `import time` nested inside the websocket-connect retry loop (line ~512
upstream), after an earlier retry loop in the same function already called `time.sleep(1)`
(line ~507). Any `import time` anywhere in a Python function makes `time` a local name for
the *whole* function body, so the earlier call raised
`UnboundLocalError: local variable 'time' referenced before assignment` on every job that
needed more than one HTTP-readiness poll. Fixed by moving `import time` to the top-level
import block and deleting the nested one. See git blame on this file for the exact diff.

`entrypoint.sh`, the four workflow JSONs and the Dockerfile are unmodified from upstream
except for the base `FROM` tag, which stays pinned to `wlsdml1114/engui_genai-base_blackwell:1.1`
as published — we do not control that image and have no Dockerfile for it. If it ever
disappears from Docker Hub, this image stops being rebuildable from scratch; our own
GHCR copy, pinned by digest, is the only durable record of the working build.

## GPU sizing

Weights loaded per job (fp8/bf16, `force_offload=True` by default so they are swapped,
not all simultaneously resident): InfiniteTalk adapter 2.7 GB, Wan2.1-I2V-14B-480P base
17 GB, lightx2v LoRA 0.7 GB, VAE 0.25 GB, umt5-xxl text encoder 6.7 GB, clip_vision_h
1.26 GB, MelBandRoformer (vocal separation) 0.46 GB. The diffusion stack alone (base +
LoRA + adapter, ~20.4 GB) must be resident during sampling. The endpoint's GPU list was
widened in the past to include the 32 GB RTX 5090; that card is cut from the list in this
repair (see ops notes) because a 32 GB card leaves too little headroom over a 20+ GB
resident model plus activations and was the likely cause of the OOM/WebSocket-closed
failure seen on `mj_1e178f88f58d`. Endpoint now targets only cards with >=48 GB.
