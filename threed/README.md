# threed — image to textured mesh

TRELLIS.2 on the ComfyUI base. Product photo in, PBR-textured GLB out.

Image: `ghcr.io/aspireco/runpod-threed-worker:latest`
Render half of the pipeline: [`threed-pipeline/`](../../threed-pipeline/)
Model comparison and licences: [`stack-docs/3D-BENCHMARK.md`](../../stack-docs/3D-BENCHMARK.md)

## Request

```json
{
  "input": {
    "workflow": "trellis2_image_to_mesh",
    "image": "<base64 PNG or JPEG>",
    "seed": 42,
    "texture_size": 4096,
    "target_faces": 700000,
    "remesh_resolution": 768
  }
}
```

Only `image` is required. Response:

```json
{
  "meshes": [
    {"node": "9001", "kind": "3d", "filename": "mos_00001_.glb",
     "bytes": 14237184, "data": "<base64 glb>"}
  ],
  "all_files": ["mos_00001_.glb", "..."],
  "prompt_id": "…"
}
```

`meshes[0]` is the full PBR mesh (UV-unwrapped, base colour + metallic + roughness +
AO + normal). `meshes[1]`, when present, is the vertex-coloured variant — the fallback
when UV unwrapping produces a poor atlas on a thin object like a shelf bracket.

Errors come back as `{"error": "<Type>: <message>"}` rather than a raised exception.

`texture_size` drives the single `PrimitiveInt` that feeds both `UnwrapMesh.resolution`
and `BakeTextureFromVoxel.texture_size`, so the UV atlas and the bake cannot disagree.
Setting `UnwrapMesh.resolution` directly would be overwritten by that link and silently
do nothing.

Easiest client is [`threed-pipeline/mesh_from_image.py`](../../threed-pipeline/mesh_from_image.py).

## What is in the image

| Model | Size | Licence | Role |
| --- | --- | --- | --- |
| TRELLIS.2 int8 | 5.25 GB | MIT | shape + texture generation |
| TRELLIS.2 shape VAE | 1.10 GB | MIT | |
| TRELLIS.2 texture VAE | 0.95 GB | MIT | |
| DINOv3 ViT-L | 1.22 GB | Meta DINOv3 Licence | image encoder — **not MIT**, see below |
| MoGe-2 | 0.66 GB | MIT | camera FOV for the TRELLIS.2 graph |
| BiRefNet | 0.44 GB | MIT | subject cut-out |

Only `trellis2_image_to_mesh` is exposed as a workflow. MoGe-2's weights are present
and drive the FOV estimate inside that graph, but the standalone *photo → scene mesh*
workflow is **not wired**: ComfyUI ships that template built around a **subgraph**, and
the build-time converter does not expand subgraphs. The template is vendored at
`workflows/moge_photo_to_scene.ui.json` for reference, and the converter raises an
error naming the problem if pointed at it. Unpack the subgraph in the ComfyUI editor
and re-export to enable it.

**DINOv3 is the one non-MIT component.** Comfy-Org's repackaged repo tags itself `mit`,
but the upstream `facebook/dinov3-*` repos are gated under Meta's own licence. Read
directly it is usable — royalty-free, worldwide, commercial use permitted, no revenue
cap, no MAU trigger, no territory exclusion — but "TRELLIS.2 is MIT" is true of
Microsoft's part, not of the whole pipeline. Full working in the benchmark doc.

## The template defaults to a different model

ComfyUI publishes this workflow as **"Pixal3D & TRELLIS.2: Image to Model"**, and the
boolean driving its three If/Else switches — titled "Boolean (Switch to Trellis2)" —
ships set to **`False`**. Out of the box it runs **Pixal3D**, not TRELLIS.2.

`build_api_workflows.py` sets it explicitly (`USE_TRELLIS2 = True`) and resolves the
switches at build time, which prunes the unselected branch and everything only it feeds
— 56 nodes down to 40, and Pixal3D's 5 GB checkpoint kept out of the image. A build-time
guard fails if a refreshed template flips the default back.

To run Pixal3D instead (also MIT, TencentARC): set `USE_TRELLIS2 = False` and add a
`fetch-model` line for
`Comfy-Org/Pixal3D/diffusion_models/pixal3d_int8_convrot.safetensors`.

## Why ComfyUI

ComfyUI **v0.34.0** ships TRELLIS.2 in core (`comfy/ldm/trellis2`,
`comfy_extras/nodes_trellis2.py`), and that is exactly what
`runpod/worker-comfyui:5.10.0-base` pins. No custom node, no compiled kernel.
TRELLIS.2's own install compiles flash-attn, nvdiffrast, a sparse-conv kernel and
diff-gaussian-rasterization against whatever CUDA it finds — the failure mode this
project has already paid for once.

**That version coupling is load-bearing.** Bumping the base image without re-running
`build_api_workflows.py` is how this breaks.

## Two traps this worker exists to avoid

**1. The stock handler returns nothing for 3D jobs.** worker-comfyui 5.10.0 collects
results with `if "images" in node_output`; ComfyUI's `SaveGLB` returns
`ui={"3d": results}`. Under the stock handler a 3D graph boots, loads 9.6 GB, samples
three times, bakes a texture atlas, writes a valid `.glb` — and returns
`{"images": []}`. A fully successful job that costs a GPU-minute and hands back
nothing, with no error. `handler.py` collects any output entry carrying a `filename`,
whatever key it arrived under.

**2. Converting the vendor template naively corrupts it silently.** ComfyUI ships these
workflows in UI format (positional `widgets_values`); `/prompt` needs API format (named
inputs). Recovering the names needs each node's `INPUT_TYPES()`, which only ComfyUI can
answer — so `build_api_workflows.py` runs **inside the image at build time**, and a
renamed input fails the build rather than a job.

The subtle part is `control_after_generate`: a seed occupies **two** slots in
`widgets_values` (the seed, then `"fixed"`). Miss it and every later widget shifts by
one — `steps` becomes `7.5`, `cfg` becomes `"euler"` — and **the graph still runs**,
producing garbage at full GPU cost. Also handled: the template wires
`ApplyTextureToMesh.base_color` from a `PreviewImage` node, which has no output slot in
any ComfyUI version; those taps resolve through to the real source.

Pruning to the ancestors of the requested outputs drops 12 of 63 nodes — five unused
preview renders and a 1024 UV-atlas render — the difference between paying for one
texture bake and six.

**3. ComfyUI ignores `--cpu` when imported as a library.** Registering nodes on a
GPU-less CI runner dies in `model_management.py`, which evaluates `get_torch_device()`
at module level. Setting `--cpu` in `sys.argv` does nothing: `cli_args.py` only calls
`parse_args()` when `comfy.options.args_parsing` is true, and that defaults to false —
`main.py` flips it on its first two lines. `enable_args_parsing()` must be called
first, and the converter asserts the flag took rather than trusting it.

## Deploy

```bash
runpodctl create endpoint --name mos-threed \
  --image ghcr.io/aspireco/runpod-threed-worker:latest \
  --gpu-type "RTX A6000" --workers-min 0 --workers-max 1 --idle-timeout 60

# PARK IT when done -- idle workers bill, and the account caps at 10 max-workers
runpodctl update endpoint <id> --workers-max 0
```

A6000 48GB ($0.53/hr) is the cheapest GPU that holds the int8 model, DINOv3, both VAEs
and the 2048 bake buffers without spilling.

## hunyuan/ — do not deploy casually

`hunyuan/` holds Hunyuan3D 2.1 as a **separate, licence-quarantined image that is
deliberately absent from the CI matrix**. Its licence excludes the European Union,
United Kingdom and South Korea — for the **outputs** as well as the model — and carries
a 1,000,000 MAU trigger. Read the header of `hunyuan/Dockerfile` before building it.
It is a benchmark comparator, not a production lane.
