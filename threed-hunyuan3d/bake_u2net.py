"""Build-time only: bake rembg's default u2net.onnx (Apache-2.0, NathanUA/U-2-Net) into the image
at the EXACT path the INSTALLED rembg version's own session class resolves, discovered by asking
that class directly rather than hardcoding a path that drifts across rembg releases.

A file, not a `RUN python -c "..."` continuation block: ACR's dependency scanner reads any line
that starts with `from` as a FROM instruction and fails the build before it begins -- this file's
own `from rembg.sessions.u2net import U2netSession` is exactly that line (confirmed live, ACR run
ch3x, 2026-10-18: "unable to understand line from rembg.sessions.u2net import U2netSession as
S;" during dependency scanning, before the build itself ever ran). Same gotcha vid-wan-a14b's
build_check.py and every other build_check.py in this repo already documents for themselves; this
script exists because the bake step needed one `from` import that a bare `-c` one-liner cannot
avoid, not because build_check.py's own rule was wrong.
"""
import os
import urllib.request

from rembg.sessions.u2net import U2netSession

d = U2netSession.model_dir()
os.makedirs(d, exist_ok=True)
dest = os.path.join(d, U2netSession.name() + ".onnx")
urllib.request.urlretrieve(
    "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx", dest
)
sz = os.path.getsize(dest)
assert sz > 150_000_000, f"u2net.onnx too small: {sz} bytes"
print("u2net.onnx baked at", dest, sz, "bytes")
