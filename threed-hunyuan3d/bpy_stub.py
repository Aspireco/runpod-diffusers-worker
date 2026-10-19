"""A minimal stand-in for `bpy` (headless Blender's Python module), installed as the actual `bpy`
module in site-packages by the Dockerfile.

Hunyuan3D-2.1's hy3dpaint/DifferentiableRenderer/mesh_utils.py does an unconditional, top-level
`import bpy` -- gating the whole module, including save_mesh()/load_mesh()/save_obj_mesh(), none
of which touch bpy's own API at all (confirmed by reading every function in that file directly).
Only convert_obj_to_glb() and its Blender-scene helpers actually call into bpy's operators, and
this worker's handler.py never calls that function -- it uses hy3dpaint.convert_utils's
pygltflib-based builder instead, matching the official model_worker.py's own choice (see
handler.py's module docstring for the full reasoning).

The real `bpy` PyPI package has no wheel for this image's Python (3.12) as of 2026-10-18: every
release from 4.2.0 through 4.5.14 ships cp311-only, and 5.2.2 ships cp313-only, with nothing for
cp312 in between (checked the PyPI JSON API directly, not assumed from the version number the
official requirements.txt pins -- "bpy==4.0" does not exist on PyPI at all). Rather than pin an
incompatible version, downgrade the whole image's Python to match a cp311 wheel, or build Blender
from source (hours, and this worker does not need Blender's actual functionality), this stub
satisfies the import and raises a clear, specific error if anything ever reaches past it into a
real Blender operator -- which would mean a future code path started needing one, and this stub
would be the wrong fix for that, not an extension of it.
"""


class _App:
    version_string = "stub (no real Blender -- see bpy.py's module docstring)"
    background = True


app = _App()


class _Unavailable:
    """Any attribute access raises -- bpy.ops.*, bpy.data.*, bpy.context.* etc. are all routed
    through instances of this, so a future call into real Blender fails loudly and specifically,
    not with a generic AttributeError three layers removed from the actual cause."""

    def __getattr__(self, name):
        def _raise(*_args, **_kwargs):
            raise RuntimeError(
                f"bpy.{name} was called, but this image ships a stub bpy (see bpy_stub.py's "
                "module docstring) with no real Blender behind it. If a code path now needs a "
                "real Blender operator, this stub is the wrong fix for it."
            )
        return _raise


ops = _Unavailable()
data = _Unavailable()
context = _Unavailable()
