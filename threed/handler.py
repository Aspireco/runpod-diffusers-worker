"""RunPod serverless handler for the 3D lane: image in, textured GLB out.

WHY THIS REPLACES worker-comfyui's OWN HANDLER
----------------------------------------------
worker-comfyui 5.10.0's handler collects results with, in effect:

    for node_id, node_output in outputs.items():
        if "images" in node_output:
            ...

ComfyUI's SaveGLB returns `IO.NodeOutput(ui={"3d": results})`. The key is "3d", not
"images". Run a 3D graph under the stock handler and it does everything right --
boots, loads 9GB of weights, samples three times, bakes a texture atlas, writes a
valid .glb to /comfyui/output -- and then returns `{"images": []}`. A completely
successful job that costs a full GPU-minute and hands back nothing, with no error
anywhere to explain it.

So this collects any output entry that *looks like a saved file* (a list of dicts
carrying "filename"), whatever key it arrived under. That also covers SaveGaussianSplat
("gaussian_splats"), SavePointCloud, and whatever key the next ComfyUI release invents.

Everything else here follows the worker-comfyui contract: ComfyUI is already running,
started by the base image's start.sh; we talk to it on 127.0.0.1:8188.
"""

print("[boot] handler starting", flush=True)

import base64
import json
import os
import time
import urllib.parse
import urllib.request

import runpod

COMFY = os.environ.get("COMFY_HOST", "127.0.0.1:8188")
BOOT_TIMEOUT = int(os.environ.get("COMFY_BOOT_TIMEOUT", "300"))
JOB_TIMEOUT = int(os.environ.get("COMFY_JOB_TIMEOUT", "1800"))
WORKFLOW_DIR = os.environ.get("WORKFLOW_DIR", "/workflows")

# Loaded at module import so FlashBoot snapshots a process that already has them --
# these are small JSON files, but the import-time rule is the house pattern and the
# failure mode it prevents (first request pays the parse) is real.
WORKFLOWS = {}
for _fn in sorted(os.listdir(WORKFLOW_DIR)) if os.path.isdir(WORKFLOW_DIR) else []:
    if _fn.endswith(".api.json"):
        with open(os.path.join(WORKFLOW_DIR, _fn), encoding="utf-8") as _fh:
            WORKFLOWS[_fn[: -len(".api.json")]] = json.load(_fh)
print(f"[boot] workflows loaded: {sorted(WORKFLOWS)}", flush=True)


def _post(path, payload):
    req = urllib.request.Request(
        f"http://{COMFY}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def _get(path, raw=False, timeout=60):
    with urllib.request.urlopen(f"http://{COMFY}{path}", timeout=timeout) as r:
        return r.read() if raw else json.loads(r.read())


def wait_for_comfy():
    deadline = time.time() + BOOT_TIMEOUT
    while time.time() < deadline:
        try:
            _get("/system_stats", timeout=5)
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"ComfyUI did not answer on {COMFY} within {BOOT_TIMEOUT}s")


def upload_image(name, b64):
    """Push the client's image into ComfyUI's input dir via /upload/image."""
    if "," in b64[:64] and b64.lstrip().startswith("data:"):
        b64 = b64.split(",", 1)[1]
    blob = base64.b64decode(b64)
    boundary = "----mos3d"
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="image"; filename="{name}"\r\n'.encode(),
        b"Content-Type: application/octet-stream\r\n\r\n",
        blob,
        f"\r\n--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n',
        f"--{boundary}--\r\n".encode(),
    ])
    req = urllib.request.Request(
        f"http://{COMFY}/upload/image",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def patch(wf, image_name, params):
    """Point LoadImage at the uploaded file and apply caller overrides.

    Patching by class_type rather than by hardcoded node id: the converter prunes the
    graph, so ids are stable only as long as the template is. Class lookup survives a
    template refresh; a hardcoded "node 122" silently patches the wrong node.
    """
    wf = json.loads(json.dumps(wf))  # deep copy: workers are reused across jobs

    loaders = [n for n in wf.values() if n["class_type"] == "LoadImage"]
    if not loaders:
        raise RuntimeError("workflow has no LoadImage node to feed")
    for n in loaders:
        n["inputs"]["image"] = image_name

    seed = params.get("seed")
    if seed is not None:
        for i, n in enumerate(v for v in wf.values() if v["class_type"] == "KSampler"):
            # Offset per sampler: the template deliberately uses different seeds per
            # stage (56 / 42 / 42). Collapsing them all to one value measurably
            # degrades the texture stage.
            n["inputs"]["seed"] = int(seed) + i

    # Input names verified against ComfyUI 0.34.0's own schemas, not guessed:
    # DecimateMesh takes `target_face_count` (not `target`), RemeshMesh takes
    # `resolution`.
    for cls, key, val in (
        ("DecimateMesh", "target_face_count", params.get("target_faces")),
        ("RemeshMesh", "resolution", params.get("remesh_resolution")),
    ):
        if val is None:
            continue
        for n in wf.values():
            if n["class_type"] == cls and key in n["inputs"]:
                n["inputs"][key] = int(val)

    # Texture size is NOT set on UnwrapMesh. In the vendor graph
    # UnwrapMesh.resolution is *wired* from a PrimitiveInt, which also feeds
    # BakeTextureFromVoxel.texture_size -- one knob driving both, so the atlas and the
    # bake can never disagree. Writing to UnwrapMesh.resolution would be overwritten by
    # the link and silently do nothing.
    tex = params.get("texture_size")
    if tex is not None:
        prims = [n for n in wf.values() if n["class_type"] == "PrimitiveInt" and "value" in n["inputs"]]
        if len(prims) != 1:
            raise RuntimeError(
                f"expected exactly one PrimitiveInt driving texture size, found {len(prims)}; "
                "refusing to guess which one texture_size means"
            )
        prims[0]["inputs"]["value"] = int(tex)

    # The SaveGLB terminals are attached at build time by build_api_workflows.py, not
    # here -- adding nodes at request time would mean a graph shape that was never
    # validated by the build. Assert they survived instead.
    if not any(n["class_type"] == "SaveGLB" for n in wf.values()):
        raise RuntimeError("workflow has no SaveGLB node; nothing would be returned")
    return wf


def collect(outputs):
    """Gather every saved file from /history, regardless of which ui key it used.

    Deliberately key-agnostic -- see the module docstring. `type == "temp"` entries are
    previews ComfyUI writes to a scratch dir and are skipped.
    """
    files = []
    for node_id, node_output in outputs.items():
        if not isinstance(node_output, dict):
            continue
        for key, entries in node_output.items():
            if not isinstance(entries, list):
                continue
            for e in entries:
                if not isinstance(e, dict) or "filename" not in e:
                    continue
                if e.get("type") == "temp":
                    continue
                q = urllib.parse.urlencode({
                    "filename": e["filename"],
                    "subfolder": e.get("subfolder", ""),
                    "type": e.get("type", "output"),
                })
                try:
                    blob = _get(f"/view?{q}", raw=True, timeout=300)
                except Exception as exc:
                    files.append({"node": node_id, "kind": key, "filename": e["filename"],
                                  "error": f"fetch failed: {exc}"})
                    continue
                files.append({
                    "node": node_id,
                    "kind": key,
                    "filename": e["filename"],
                    "bytes": len(blob),
                    "data": base64.b64encode(blob).decode(),
                })
    return files


def run(workflow, image_b64, params):
    wait_for_comfy()
    name = f"mos_input_{int(time.time() * 1000)}.png"
    upload_image(name, image_b64)

    wf = patch(workflow, name, params)
    res = _post("/prompt", {"prompt": wf})
    if "prompt_id" not in res:
        raise RuntimeError(f"ComfyUI rejected the graph: {json.dumps(res)[:900]}")
    pid = res["prompt_id"]
    print(f"[job] queued {pid}", flush=True)

    deadline = time.time() + JOB_TIMEOUT
    while time.time() < deadline:
        hist = _get(f"/history/{pid}")
        entry = hist.get(pid)
        if entry is not None:
            status = entry.get("status") or {}

            # ComfyUI creates the history entry when execution STARTS and fills in
            # `outputs` node by node as the graph runs, so the entry existing proves
            # nothing about being finished. `completed` must be explicitly true.
            #
            # The first version read `not status.get("completed", True)` -- defaulting a
            # MISSING flag to "done". That turned an in-progress graph into a finished
            # one: the handler read history ~6s in, when only the cheap preview nodes had
            # produced output, found no mesh among them and declared failure on a job
            # that was running correctly. The graph was never at fault. Defaulting an
            # unknown state to "success" is how a race becomes a bug report.
            if status.get("status_str") == "error":
                msgs = status.get("messages", [])
                raise RuntimeError(f"graph failed: {json.dumps(msgs)[:1200]}")
            if status.get("completed") is not True:
                time.sleep(2)
                continue

            files = collect(entry.get("outputs", {}))
            meshes = [f for f in files if f["filename"].lower().endswith((".glb", ".gltf", ".obj", ".ply"))]
            if not meshes:
                # Carry the whole picture, not just the symptom. "No mesh" has several
                # very different causes -- an output node that never ran, one that ran
                # and wrote nothing, or a silent execution error ComfyUI still reports
                # as completed -- and they are indistinguishable from the outputs dict
                # alone. Each round trip here costs a rebuild plus a cold start, so
                # spend the bytes once rather than guessing twice.
                outs = entry.get("outputs", {})
                submitted = {}
                for n_id, n in wf.items():
                    submitted.setdefault(n["class_type"], []).append(n_id)
                saves = {n_id: n["inputs"] for n_id, n in wf.items()
                         if n["class_type"] in ("SaveGLB", "Save3DAdvanced", "MeshToFile3D")}
                raise RuntimeError(json.dumps({
                    "error": "graph completed but produced no mesh file",
                    "outputs_seen": {k: {kk: len(vv) if isinstance(vv, list) else str(vv)[:60]
                                         for kk, vv in v.items()} for k, v in outs.items()},
                    "status": entry.get("status"),
                    "save_nodes_submitted": saves,
                    "node_classes_submitted": {k: v for k, v in sorted(submitted.items())},
                    "prompt_id": pid,
                })[:3500])
            return {"meshes": meshes, "all_files": [f["filename"] for f in files], "prompt_id": pid}
        time.sleep(2)
    raise RuntimeError(f"job {pid} exceeded {JOB_TIMEOUT}s")


def handler(job):
    try:
        job_input = job.get("input") or {}
        name = job_input.get("workflow", "trellis2_image_to_mesh")
        if name not in WORKFLOWS:
            return {"error": f"unknown workflow {name!r}; have {sorted(WORKFLOWS)}"}
        image = job_input.get("image")
        if not image:
            return {"error": "input.image is required (base64-encoded PNG or JPEG)"}
        return run(WORKFLOWS[name], image, job_input)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


runpod.serverless.start({"handler": handler})
