"""Teach worker-comfyui's handler to return audio. Applied at build time.

THE BUG THIS FIXES. worker-comfyui 5.10.0's handler walks exactly one output key:

    for node_id, node_output in outputs.items():
        if "images" in node_output:
            ...
    other_keys = [k for k in node_output.keys() if k != "images"]   # -> a print, nothing more

ComfyUI's SaveAudio / SaveAudioMP3 / SaveAudioOpus nodes emit their results under the
"audio" key, not "images". So a TTS graph runs to completion, ComfyUI writes a perfectly
good .flac to its output directory, and the handler logs "unhandled output keys: ['audio']"
and returns `{}`. The job is reported SUCCESSFUL with no audio in it. That is the worst
possible failure mode -- a green tick, a billed GPU-minute, and nothing to show for it --
and it would not have surfaced until the first real generation.

THE FIX. Everything downstream of the key check is already format-agnostic: ComfyUI's
/view endpoint serves any output file, and the returned extension is taken from the
filename (`os.path.splitext(filename)[1]`), so .flac/.mp3/.opus/.wav all survive the trip.
Only the key name is wrong. So we normalise before the loop -- fold any "audio" entries
into "images" -- and leave the 100 lines of S3/base64/error handling untouched. A rewrite
of that loop would be a bigger diff with more to go wrong on the next base-image bump.

The output key in the response therefore stays "images" even for audio. Deliberate: the
alternative is forking the response schema, and every RunPod example, every retry helper
and every downstream consumer in this repo already reads `output.images[]`. One vocabulary
beats two.

Idempotent, and asserts loudly. If a future base image renames the handler, changes the
anchor line, or fixes this upstream, the build fails here with a readable message rather
than producing an image that silently drops audio again.
"""

import re
import sys

HANDLER = "/handler.py"

# Anchor on the line that materialises the outputs dict -- the last point at which we can
# still reshape every node's output before the key check runs.
ANCHOR = 'outputs = prompt_history.get("outputs", {})'

PATCH = '''
        # --- audio passthrough (patched at build time; see patch_audio_outputs.py) ---
        # SaveAudio* nodes report under "audio"; the loop below only knows "images".
        # Fold one into the other so audio takes the existing /view + base64/S3 path.
        for _node_out in outputs.values():
            for _key in ("audio", "gifs"):
                _extra = _node_out.pop(_key, None)
                if _extra:
                    _node_out.setdefault("images", []).extend(_extra)
                    print(
                        f"worker-comfyui - folded {len(_extra)} '{_key}' output(s) "
                        "into images for return"
                    )
        # --- end audio passthrough ---
'''

src = open(HANDLER, encoding="utf-8").read()

if "audio passthrough" in src:
    print("[patch] already applied, nothing to do")
    sys.exit(0)

if src.count(ANCHOR) != 1:
    sys.exit(
        f"[patch] FAILED: expected exactly 1 occurrence of anchor {ANCHOR!r}, "
        f"found {src.count(ANCHOR)}. The base image's handler changed -- re-derive "
        "the patch against the new worker-comfyui before shipping this image."
    )

src = src.replace(ANCHOR, ANCHOR + "\n" + PATCH.strip("\n"), 1)
open(HANDLER, "w", encoding="utf-8").write(src)

# Prove it parses and that the fold really is upstream of the key check, rather than
# trusting that a string replace landed where we meant it to.
import ast

ast.parse(src)
fold_at = src.index("audio passthrough")
check_at = src.index('if "images" in node_output:')
assert fold_at < check_at, "[patch] fold landed AFTER the images check -- would be a no-op"
print("[patch] audio passthrough applied and verified")
