# threed-hunyuan3d — Hunyuan3D-2.1 (image to textured mesh)

Image in, PBR-textured GLB out. Azure GPU fleet lane `hunyuan3d`, A100 80GB PCIe. A second,
licence-distinct opinion alongside `workers/threed` (TRELLIS.2, MIT) — see the licence table
below for why this one needs its own environment.

## Premade search

| Need | Searched | Using | Why |
|---|---|---|---|
| A ready-made worker/server | `j3store-studio/Hunyuan3D-2.1-ONNX-cached` (a RunPod serverless worker on GitHub, found via web search) | **Not used** | It optimises for RunPod cold-start specifically (ONNX export + RunPod Model Caching) and is a third party's reshaping of the model, not the official code. The task asked for the official code paths directly; a from-scratch Azure port of an already-reshaped third-party worker would add a second layer of reinterpretation on top of Tencent's own, for no benefit here (Azure's weight-staging model is already different from RunPod's Model Caching, so none of that worker's cold-start optimisation carries over anyway). |
| A ComfyUI node/workflow | `kijai/ComfyUI-Hunyuan3DWrapper` (1k+ stars, the de facto ComfyUI path for Hunyuan3D, incl. the texture stage) | **Not used** | A real option and arguably the most battle-tested community path, but it is still a reimplementation of Tencent's pipeline as ComfyUI nodes. `workers/threed/hunyuan/` already tried the OTHER ComfyUI route (core ComfyUI's own Hunyuan3D 2.1 nodes) and documented that it is geometry-only in core — no texture stage — which is why that folder is quarantined as "a benchmark comparator, not a production lane." The task explicitly asked for the official repo's own pipeline classes; going through either ComfyUI path here would be a step away from that, not toward it. |
| The official repo's own server | `Tencent-Hunyuan/Hunyuan3D-2.1`'s `api_server.py` / `model_worker.py` / `gradio_app.py` | **Read, not run** — `model_worker.py`'s exact generation sequence and OBJ→GLB conversion choice are followed (see handler.py's module docstring), but the FastAPI/Gradio server code itself is not used; this worker has its own `handler.py` under the Azure lane contract | `api_server.py` is a long-running FastAPI process with its own polling `/status/{uid}` protocol — the wrong shape for a queue-drained batch job. Its *generation logic* (`model_worker.py`) is exactly the official code path the task asked for, so that part is followed closely; the HTTP server wrapper around it is not needed. |
| Official shape pipeline | `hy3dshape/hy3dshape/pipelines.py` | **`Hunyuan3DDiTFlowMatchingPipeline`**, vendored at the pinned commit, used through its public `from_pretrained`/`__call__` API | Exactly what the task named. No reimplementation. |
| Official texture pipeline | `hy3dpaint/textureGenPipeline.py` | **`Hunyuan3DPaintPipeline`**, same vendoring | Exactly what the task named — the full multiview-diffusion + Real-ESRGAN + custom_rasterizer PBR bake, not the texture-less ComfyUI core path. |

Nothing here was built from scratch except `handler.py` and `build_check.py` (glue to the Azure
lane contract, not model code) and this README/manifest set. The model, its weights, and its
shape + texture pipelines are the official project's, pinned to commit
`82920d643c0dc2f7bfd7255f45f62d386edfe60c` (HEAD of `main`, 2026-10-18).

## Licence — read before production use

| Component | Source | Licence | Territory / output restriction |
|---|---|---|---|
| Hunyuan3D-2.1 code (shape + paint pipelines, custom_rasterizer, DifferentiableRenderer) | `github.com/Tencent-Hunyuan/Hunyuan3D-2.1` @ `82920d6` | **Tencent Hunyuan 3D 2.1 Community License** | Territory = worldwide **excluding the EU, UK and South Korea** (LICENSE S1(l)) |
| Hunyuan3D-2.1 weights (`hunyuan3d-dit-v2-1`, `hunyuan3d-paintpbr-v2-1`, `hunyuan3d-vae-v2-1`) | `huggingface.co/tencent/Hunyuan3D-2.1` | Same licence | Same Territory |
| Generated meshes (Outputs) | — | Same licence, **S5(c)** | "You must not use, reproduce, modify, distribute, or display the Tencent Hunyuan 3D 2.1 Works, Output or results of the Tencent Hunyuan 3D 2.1 Works outside the Territory." This is why the image is built only in `azswmcacr` (East US 2, inside the Territory) and runs only in the Canada Central GPU environment. |
| Training other models on Outputs | — | Same licence, **S5(b)** | "You must not use the Tencent Hunyuan 3D 2.1 Works or any Output or results ... to improve any other AI model (other than Tencent Hunyuan 3D 2.1 or Model Derivatives thereof)." |
| Scale threshold | — | Same licence, **S4** | Above **1,000,000** monthly active users (all products/services, combined), a separate licence must be requested from Tencent. **Two orders of magnitude lower** than the 100,000,000 the companion `vid-hunyuan-i2v` lane's Tencent Hunyuan VIDEO licence uses (same S1(l)/S4/S5(b)/S5(c) structure, different number) — confirmed by fetching both LICENSE files directly and diffing them, not assumed from either one. |
| DINOv2 (`facebook/dinov2-giant`) image encoder | Meta, via Hugging Face | **Apache-2.0** | No territory or MAU restriction — clean |
| Real-ESRGAN x4plus checkpoint | `xinntao/Real-ESRGAN` | **BSD-3-Clause** | Clean |
| rembg's default model (`u2net.onnx`) | `NathanUA/U-2-Net`, via `danielgatis/rembg` | **Apache-2.0** | Clean |
| `bpy` (headless Blender Python module) | Blender Foundation | GPL-licensed upstream (the `bpy` PyPI distribution); imported for module-load reasons only — see "bpy is a real dependency" below, this worker's own OBJ→GLB conversion does not call into its operators | Not redistributed as a combined work beyond what Tencent's own official code already does |

All four Tencent-licence clauses (S1(l), S4, S5(b), S5(c)) verified by fetching
`Tencent-Hunyuan/Hunyuan3D-2.1`'s `LICENSE` file directly on 2026-10-18, not summarised from
memory or from the task brief.

## Handler contract

Input (`job["input"]`):

```json
{
  "image": "data:image/png;base64,... or raw base64 or https://...",
  "remove_background": true,
  "texture": true,
  "target_faces": 40000,
  "seed": 1234,
  "steps": 50,
  "octree_resolution": 384,
  "guidance_scale": 5.0
}
```

- `image`: data URL, raw base64, or an `https://` URL. Required.
- `remove_background` (default **true**): runs `hy3dshape.rembg.BackgroundRemover` (u2net) before
  shape generation. See "Known deviations" below — the official demo/server code has a bug that
  means this never actually ran upstream; it is wired for real here.
- `texture` (default **true**): runs the full PBR paint pipeline after shape generation. On
  failure, falls back to the untextured mesh (`textured: false` in the response) rather than
  failing the whole job — matching `model_worker.py`'s own try/except around texturing.
- `target_faces` (default 40000, matching `FaceReducer`'s own class default and the official
  `api_models.py`'s declared default): applied to the untextured mesh via `FaceReducer` before
  texturing. **When `texture: true`, the paint stage's own internal remesh step
  (`use_remesh=True`, its own default) re-simplifies to its own hardcoded 40000-face target
  regardless of this value** — see "Known deviations".
- `steps` (default 50) and `octree_resolution` (default 384): the shape pipeline's own library
  defaults (`Hunyuan3DDiTFlowMatchingPipeline.__call__`'s own defaults), not `api_server.py`'s
  declared Pydantic defaults (5 and 256) — see "Known deviations" for why.
- `guidance_scale` (default 5.0): both the library's own default and `api_server.py`'s declared
  default agree on this one.
- `seed` (default 1234, matching `api_models.py`): deterministic by default, unlike the bare
  `demo.py`, which passes no generator at all.

Output:

```json
{
  "meshes": [{"filename": "textured.glb", "bytes": 8345213,
              "data": "data:model/gltf-binary;base64,..."}],
  "all_files": ["textured.glb"],
  "faces": 39872,
  "vertices": 21004,
  "textured": true,
  "seed": 1234,
  "steps": 50,
  "octree_resolution": 384,
  "generation_seconds": 184.2,
  "paint_seconds": 96.5,
  "model": "tencent/Hunyuan3D-2.1",
  "licence": "Tencent Hunyuan 3D 2.1 Community (Territory excludes EU/UK/South Korea; binds Outputs; 1M MAU threshold)"
}
```

`meshes[0]` mirrors the `threed` (TRELLIS.2) lane's own mesh-entry shape (`filename`/`bytes`/`data`
with a `data:model/gltf-binary;base64,...` URL) so a client written against one 3D lane reads the
other the same way; `faces`/`vertices`/`generation_seconds` are added on top, per the task brief.
`all_files` is kept for the same reason even though this lane only ever returns one mesh (TRELLIS.2's
own `all_files` can list intermediates it decided not to return). ComfyUI-specific fields from the
TRELLIS.2 shape (`node`, `kind`, `prompt_id`) are not reproduced — they describe ComfyUI graph
internals this lane has none of, and inventing meaningless placeholder values for them would be
worse than omitting them.

On failure: `{"error": "<Type>: <message>"}`, never a raised exception.

## Known deviations from the official demo/server code (read before trusting this blind)

- **`remove_background` is fixed, not just passed through.** Both `demo.py` and `model_worker.py`
  write `image = image.convert("RGBA")` and only THEN check `if image.mode == "RGB":` before
  running background removal — a check that can never be true, since the line above already
  forced the mode to `"RGBA"`. Confirmed by reading both files directly; both ship the identical
  ordering mistake, so background removal has never actually executed through either official
  entry point. `handler.py` checks the job's own `remove_background` flag directly instead.
- **`steps`/`octree_resolution` use the library's own defaults (50 / 384), not
  `api_models.py`'s declared Pydantic defaults (5 / 256).** `model_worker.py`'s
  `ModelWorker.generate()` calls `self.pipeline(image=image)[0]` with **no other kwargs at
  all** — every field `GenerationRequest` declares (`num_inference_steps`, `octree_resolution`,
  `guidance_scale`, `num_chunks`, `face_count`, `seed`) except the image itself is parsed into
  Pydantic and then silently never read. So the api_server's declared "defaults" are not what the
  official server actually runs today; the library's own `__call__` defaults are. This handler
  wires all of these knobs through for real (they are declared in the input schema for a reason),
  using the library's own defaults as the baseline rather than the server's unused ones, and notes
  the discrepancy here rather than picking one arbitrarily.
- **`target_faces` is real for the untextured mesh, partially overridden for the textured one.**
  `Hunyuan3DPaintPipeline.__call__`'s own `use_remesh=True` default (which this handler does not
  override) calls `simplify_mesh_utils.remesh_mesh()`, which hardcodes `target_count=40000` with
  no parameter to change it. A `target_faces` far below 40000 will still visibly reduce the
  untextured mesh (and therefore the geometry the paint stage starts from), but the FINAL textured
  mesh's face count converges toward ~40000 regardless. The response's `faces`/`vertices` are
  always read back from the actual exported file, never assumed, so this shows up honestly in the
  response rather than silently claiming the requested number.
- **OBJ→GLB conversion uses `hy3dpaint.convert_utils.create_glb_with_pbr_materials`
  (pygltflib), not `DifferentiableRenderer.mesh_utils.convert_obj_to_glb` (Blender/`bpy`).**
  This is `model_worker.py`'s own choice (`save_glb=False` + its own `quick_convert_with_obj2gltf`
  helper), not an invention of this worker — see handler.py's module docstring for the full
  reasoning (OBJ's MTL format has no standard PBR metallic/roughness slot, so the bpy path's
  fidelity there depends on Blender's own OBJ importer). `bpy` is still a real, required *import*
  dependency regardless: `DifferentiableRenderer/mesh_utils.py`'s top-level `import bpy` gates
  the whole module, and `MeshRender.save_mesh()` — called unconditionally inside the paint
  pipeline, not behind `save_glb` — delegates to that module's free `save_mesh()` function. Skip
  installing `bpy` and the paint stage breaks even though nothing calls `bpy`'s own operators.
- **DINOv2 is fetched because the code reading confirms it is needed, not defensively.**
  `hunyuanpaintpbr/unet/modules.py`'s `UNet2p5DConditionModel.__init__` hardcodes
  `self.use_dino = True` (line 788 at the pinned commit), and `multiviewDiffusionNet.__init__`
  only builds the `Dino_v2` wrapper `if ... self.pipeline.unet.use_dino`. `facebook/dinov2-giant`
  (Apache-2.0) is therefore on the call path, not an optional extra — its own manifest and
  directory, per the task's own instruction for exactly this situation.
- **The standalone `hunyuan3d-vae-v2-1/` weights are fetched but not currently loaded by this
  handler.** `Hunyuan3DDiTPipeline.from_single_file` reads the VAE's state dict out of the SAME
  `hunyuan3d-dit-v2-1/model.fp16.ckpt` file (`ckpt['vae']`), confirmed by reading that method
  directly — the separate `hunyuan3d-vae-v2-1/model.fp16.ckpt` is only reached by
  `pipeline.enable_flashvdm()` (a turbo-VAE swap this handler does not call). Fetched anyway to
  match the task brief's own 14.9GB figure and in case a future tuning pass turns flashvdm on.
- **Could not verify `DiffusionPipeline.from_pretrained(..., custom_pipeline=..., torch_dtype=
  torch.float16)` actually resolves `model_index.json`'s `"unet": ["modules",
  "UNet2p5DConditionModel"]` entry correctly against the vendored `hunyuanpaintpbr/` directory —
  this needs the real 14.9GB of weights (not present on the CPU-only ACR build agent, and out of
  scope to download during the build just to test) and diffusers' own dynamic custom-pipeline
  component-loading mechanism, which this worker does not reimplement or second-guess in any way.
  `build_check.py`'s `paint` check verifies `Hunyuan3DPaintConfig`/`Hunyuan3DPaintPipeline`'s
  public shape instead — the right-sized check for a no-GPU, no-real-weights build verification.
  This is the one piece of this worker I would check first against a real job if texturing comes
  back looking wrong.

## VRAM reasoning on an A100 80GB (from the official model card, not benchmarked here)

Tencent's own `README.md` for the repo states VRAM requirements directly: **~10GB for shape
generation, ~21GB for texture generation, ~29GB for both in one process** — comfortably inside an
A100 80GB with more than 2.5x headroom, which is consistent with `Hunyuan3DPaintConfig` hardcoding
`device = "cuda"` with no offload knobs anywhere in the official code (there would be no need to
add any if 29GB was already the expectation on much smaller cards). No GPU job was started to
confirm this independently; the task scope excludes it.

## Files under `$HUNYUAN3D_DIR` (default `/runpod-volume/hunyuan3d`) and `$HUNYUAN3D_DINOV2_DIR`

`gpu/manifests/hunyuan3d.tsv` (22 files, 14.91 GB) mirrors `tencent/Hunyuan3D-2.1`'s own top-level
folder layout exactly (its `hy3dpaint/` subfolder — a duplicate, older copy of two Python source
files already vendored from GitHub — and the two READMEs are excluded; nothing in the handler's
import path reads them). `gpu/manifests/hunyuan3d-dinov2.tsv` (3 files, 4.55 GB) mirrors
`facebook/dinov2-giant`, trimmed to the `safetensors` weight only (its duplicate `pytorch_model.bin`
is excluded — same "one format, not both" trim `vid-wan-animate`'s README documents for its own
CLIP tokenizer).

```
$HUNYUAN3D_DIR/
├── hunyuan3d-dit-v2-1/{config.yaml, model.fp16.ckpt}                       # shape DiT + its own VAE + conditioner, one ckpt
├── hunyuan3d-paintpbr-v2-1/                                                 # the full diffusers-format paint pipeline
│   ├── model_index.json
│   ├── feature_extractor/preprocessor_config.json
│   ├── image_encoder/{config.json, model.safetensors}                      # CLIP vision
│   ├── scheduler/scheduler_config.json
│   ├── text_encoder/{config.json, pytorch_model.bin}                       # CLIP text
│   ├── tokenizer/{merges.txt, special_tokens_map.json, tokenizer_config.json, vocab.json}
│   └── unet/{attn_processor.py, config.json, diffusion_pytorch_model.bin, model.py, modules.py}
└── hunyuan3d-vae-v2-1/{config.yaml, model.fp16.ckpt}                        # NOT on the default call path -- see "Known deviations"

$HUNYUAN3D_DINOV2_DIR/
├── config.json
├── preprocessor_config.json
└── model.safetensors
```

Wire the fetch job with repo id `tencent/Hunyuan3D-2.1` and destination `$HUNYUAN3D_DIR` against
`gpu/manifests/hunyuan3d.tsv`, and a second fetch with repo id `facebook/dinov2-giant` and
destination `$HUNYUAN3D_DINOV2_DIR` against `gpu/manifests/hunyuan3d-dinov2.tsv`.

## Build

```bash
MSYS_NO_PATHCONV=1 PYTHONIOENCODING=utf-8 "/c/Program Files/Microsoft SDKs/Azure/CLI2/python.exe" -IBm azure.cli acr build \
  -r azswmcacr -t hunyuan3d-worker:<tag> \
  "workers/threed-hunyuan3d" \
  --no-logs --timeout 7200 --query runId -o tsv
```

`<tag>` = the short git sha of this repo's HEAD. Expect a long build: cloning the repo, compiling
two native extensions, and installing `bpy` (a large wheel) all add up, even with no GPU work.

## Run

Through the Azure runner shim (`gpu/azure_runner.py`, applied on top of this image by
`gpu/Dockerfile.layer`), which imports this file's `handler.py` and feeds it jobs from Storage
Queue `gpu-hunyuan3d`. Not run directly — `python handler.py` tries to import `runpod`'s
serverless loop with no queue behind it.

## Files

| file | what it is |
|---|---|
| `Dockerfile` | the image; built on ACR as `azswmcacr.azurecr.io/hunyuan3d-worker`; vendors the official repo at `/opt/hunyuan3d-src`, compiles `custom_rasterizer` + `mesh_inpaint_processor`, bakes in RealESRGAN + u2net |
| `handler.py` | the serverless worker: boot-time weight check, offline `snapshot_download` redirect, shape + face-reduce + paint + pygltflib GLB export |
| `build_check.py` | build-time checks: every pip import, the shape/paint API shape, the two compiled extensions |
