"""Run at image build time: fail the BUILD, not a billed GPU.

Runs handler.py's own module-level construction for real, once per MODE value the deployed
image is switched between (MODE=upscale loads the two baked Real-ESRGAN checkpoints via
spandrel; MODE=interpolate loads the baked RIFE v4.26 checkpoint) -- the exact same code,
weights and assertions handler.py itself runs at a real cold start, not a hand-copied
snippet that could quietly drift out of sync with it. Both branches already fall back to
CPU when no GPU is present (`DEVICE = "cuda" if torch.cuda.is_available() else "cpu"`), which
is true on this GPU-less ACR build agent, so this is safe to run here -- confirmed by reading
handler.py's own module scope, not assumed.

handler.py ends with `runpod.serverless.start({"handler": handler})`, which would otherwise
block waiting for jobs forever. Patched out first, exactly as gpu/azure_runner.py's own
load_handler() does to import a worker safely -- this mirrors that mechanism rather than
inventing a second one.

A file, not an inline Dockerfile RUN: this used to be two separate `python -c "..."` checks
directly in the Dockerfile (one for spandrel, one for RIFE). The RIFE one broke ACR's build:
its second physical line (after a `\` continuation) was
    from train_log.IFNet_HDv3 import IFNet;
and ACR's dependency scanner reads a continuation line starting with `from` as an attempted
FROM instruction, failing the build before Docker itself ever runs it (the identical failure
tts/build_check.py's own docstring records for a different lane).
"""
import os

import runpod.serverless as _rs

_rs.start = lambda config: None  # handler.py's own call becomes a no-op; see docstring above

import runpy  # noqa: E402

for _mode in ("upscale", "interpolate"):
    os.environ["MODE"] = _mode
    print(f"[build_check] running handler.py module scope with MODE={_mode}", flush=True)
    runpy.run_path("/handler.py", run_name="__main__")

print("[build_check] upscale (Real-ESRGAN x2/x4) + interpolate (RIFE v4.26) construct OK")
