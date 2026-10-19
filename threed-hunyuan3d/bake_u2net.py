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

The path-resolution call itself tries two API shapes rather than assuming one: the FIRST attempt
at this script (ACR run ch42, 2026-10-18) called `U2netSession.model_dir()`, researched by
reading rembg's GitHub `main` branch -- but the pinned release actually installed here,
`rembg==2.0.65`, predates that method entirely ("AttributeError: type object 'U2netSession' has
no attribute 'model_dir'", build failed before the real work even started). Checked the ACTUAL
tagged source for 2.0.65 this time (`github.com/danielgatis/rembg/blob/v2.0.65/rembg/sessions/
{u2net,base}.py`), which saves straight to `<u2net_home>/<name>.onnx` via `cls.u2net_home()`, no
`model_dir()` subfolder nesting. Trying `model_dir()` first and falling back to `u2net_home()`
means this keeps working if the pin is ever bumped to a release with the newer layout too.
"""
import os
import urllib.request

from rembg.sessions.u2net import U2netSession

if hasattr(U2netSession, "model_dir"):
    d = U2netSession.model_dir()
    dest = os.path.join(d, U2netSession.name() + ".onnx")
else:
    d = U2netSession.u2net_home()
    dest = os.path.join(d, U2netSession.name() + ".onnx")
os.makedirs(d, exist_ok=True)
urllib.request.urlretrieve(
    "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx", dest
)
sz = os.path.getsize(dest)
assert sz > 150_000_000, f"u2net.onnx too small: {sz} bytes"
print("u2net.onnx baked at", dest, sz, "bytes")
