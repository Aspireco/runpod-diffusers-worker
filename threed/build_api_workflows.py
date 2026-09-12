"""Convert ComfyUI UI-format workflow templates to API format, at BUILD time.

WHY THIS RUNS IN THE DOCKERFILE AND NOT AT RUNTIME
--------------------------------------------------
The templates under workflows/ are the ones ComfyUI itself ships -- their sampler
counts, CFG values and remesh/decimate targets are the vendor's, not ours. But they
are saved in the *UI* format the web editor uses, and /prompt only accepts the *API*
format. The two differ in a way that cannot be bridged by a static mapping:

    UI  : widget values are a positional list, `widgets_values`, with no names
    API : every input is named

Recovering the names needs each node's INPUT_TYPES(), which only ComfyUI can answer.
So the conversion runs inside the image, against the exact ComfyUI that will execute
the graph. If a node was renamed, lost an input, or never existed in this version,
this script raises and the *build* fails with a readable traceback -- instead of the
graph failing on a GPU that bills by the second and whose only symptom would be a
worker returning `{}`.

THE FOUR THINGS THAT MAKE THIS NON-TRIVIAL
------------------------------------------
1. `control_after_generate`. A seed widget occupies TWO slots in widgets_values --
   the seed, then the control mode ("fixed"/"randomize"). KSampler serialises as
   [56,"fixed",12,7.5,"euler","normal",1] but has six API inputs. Consume the extra
   slot or every value after the seed lands on the wrong input, silently, and the
   graph still runs -- producing garbage at full GPU cost.

2. Output nodes used as sources. The TRELLIS.2 template wires
   `ApplyTextureToMesh.base_color` from a PreviewImage node. PreviewImage has no
   output slots in any ComfyUI version -- the editor allows it as a visual tap.
   Resolved by walking through to whatever feeds the PreviewImage.

3. Muted (mode 2) and bypassed (mode 4) nodes. Bypass means "pass input through";
   mute means "do not execute". Both are rewritten, not ignored.

4. Dead branches. The template renders five preview images we never look at. They
   are pruned to the ancestors of the requested output nodes, which is the
   difference between paying for one texture bake and paying for six.
"""

import json
import os
import sys

sys.path.insert(0, "/comfyui")

WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}


def load_node_defs():
    """Return ComfyUI's NODE_CLASS_MAPPINGS with all extras registered."""
    import nodes

    init = getattr(nodes, "init_extra_nodes", None)
    if init is not None:
        try:
            res = init(init_custom_nodes=True)
        except TypeError:
            res = init()
        if hasattr(res, "__await__"):  # newer ComfyUI made this a coroutine
            import asyncio

            asyncio.run(res)
    return nodes.NODE_CLASS_MAPPINGS


def input_spec(cls):
    """Ordered [(name, type, options)] for a node class, required then optional."""
    it = cls.INPUT_TYPES()
    out = []
    for section in ("required", "optional"):
        for name, spec in (it.get(section) or {}).items():
            if isinstance(spec, (list, tuple)) and spec:
                typ = spec[0]
                opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            else:
                typ, opts = spec, {}
            out.append((name, typ, opts))
    return out


def is_widget(typ, opts):
    """True when a value is typed into the node rather than wired into it."""
    if opts.get("forceInput"):
        return False
    if isinstance(typ, (list, tuple)):  # a combo box -- its options are the list
        return True
    return typ in WIDGET_TYPES


def has_outputs(cls):
    return bool(getattr(cls, "RETURN_TYPES", ()) or [])


class Graph:
    def __init__(self, ui, node_defs):
        self.defs = node_defs
        self.nodes = {n["id"]: n for n in ui["nodes"] if n["type"] not in ("Note", "MarkdownNote")}
        # link id -> (origin_node_id, origin_slot)
        self.links = {}
        for l in ui.get("links") or []:
            if isinstance(l, dict):
                self.links[l["id"]] = (l["origin_id"], l["origin_slot"])
            else:
                self.links[l[0]] = (l[1], l[2])

    def resolve(self, link_id, _seen=None):
        """Follow a link back to a node that can actually be an API source.

        Walks through bypassed nodes (mode 4) and through output-only nodes such as
        PreviewImage that the editor permits as a visual tap but that have no real
        output slot. Returns (node_id, slot) or None if the chain dead-ends.
        """
        _seen = _seen or set()
        if link_id is None or link_id in _seen:
            return None
        _seen.add(link_id)

        origin = self.links.get(link_id)
        if origin is None:
            return None
        nid, slot = origin
        node = self.nodes.get(nid)
        if node is None:
            return None

        cls = self.defs.get(node["type"])
        mode = node.get("mode", 0)
        passthrough = mode in (2, 4) or (cls is not None and not has_outputs(cls))
        if not passthrough:
            return (nid, slot)

        # Pass through: prefer an upstream input whose type matches this slot, else
        # the first connected input.
        want = None
        outs = node.get("outputs") or []
        if slot < len(outs):
            want = outs[slot].get("type")
        candidates = [i for i in (node.get("inputs") or []) if i.get("link") is not None]
        for i in candidates:
            if want is not None and i.get("type") == want:
                return self.resolve(i["link"], _seen)
        return self.resolve(candidates[0]["link"], _seen) if candidates else None

    def to_api(self, keep_ids):
        """Emit API-format prompt containing keep_ids and all their ancestors."""
        needed, stack = set(), list(keep_ids)
        while stack:
            nid = stack.pop()
            if nid in needed or nid not in self.nodes:
                continue
            needed.add(nid)
            for i in self.nodes[nid].get("inputs") or []:
                src = self.resolve(i.get("link"))
                if src:
                    stack.append(src[0])

        api = {}
        for nid in sorted(needed):
            node = self.nodes[nid]
            ntype = node["type"]
            cls = self.defs.get(ntype)
            if cls is None:
                raise SystemExit(
                    f"node type {ntype!r} (id {nid}) is not installed in this ComfyUI -- "
                    "the template needs a newer core or a custom node the image lacks"
                )
            if node.get("mode", 0) in (2, 4):
                continue  # muted/bypassed: never emitted, callers reach through it

            linked = {i["name"]: i["link"] for i in (node.get("inputs") or []) if i.get("link") is not None}
            wv = list(node.get("widgets_values") or [])
            wi = 0
            inputs = {}

            for name, typ, opts in input_spec(cls):
                if name in linked:
                    src = self.resolve(linked[name])
                    if src is None:
                        raise SystemExit(
                            f"{ntype}#{nid}.{name}: link {linked[name]} dead-ends "
                            "(muted node with nothing upstream?)"
                        )
                    inputs[name] = [str(src[0]), src[1]]
                    continue
                if not is_widget(typ, opts):
                    continue
                if wi >= len(wv):
                    continue  # optional widget the template left at its default
                inputs[name] = wv[wi]
                wi += 1
                # The seed's companion "fixed"/"randomize" slot. Skipping this is the
                # single most damaging bug possible here: every later widget shifts by
                # one and the graph runs anyway, producing wrong output at full cost.
                if opts.get("control_after_generate") and wi < len(wv) and isinstance(wv[wi], str):
                    wi += 1

            api[str(nid)] = {"class_type": ntype, "inputs": inputs, "_meta": {"title": node.get("title", ntype)}}
        return api


def convert(ui_path, out_path, keep_ids, node_defs):
    with open(ui_path, encoding="utf-8") as fh:
        ui = json.load(fh)
    g = Graph(ui, node_defs)

    missing = [i for i in keep_ids if i not in g.nodes]
    if missing:
        raise SystemExit(f"{ui_path}: requested output node ids {missing} are not in the template")

    api = g.to_api(keep_ids)

    # Every reference must point at a node we actually emitted.
    for nid, node in api.items():
        for name, val in node["inputs"].items():
            if isinstance(val, list) and len(val) == 2 and isinstance(val[0], str):
                if val[0] not in api:
                    raise SystemExit(
                        f"{ui_path}: {node['class_type']}#{nid}.{name} references node "
                        f"{val[0]}, which was pruned -- the keep set is wrong"
                    )
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(api, fh, indent=1)
    print(f"[build] {os.path.basename(ui_path)} -> {os.path.basename(out_path)}: "
          f"{len(api)} nodes (from {len(g.nodes)})")
    return api


def main():
    defs = load_node_defs()
    print(f"[build] ComfyUI exposes {len(defs)} node types")

    # Single-template mode, used by the quarantined Hunyuan3D image:
    #   python build_api_workflows.py <in.ui.json> <out.api.json> <keep_id,keep_id,...>
    if len(sys.argv) == 4:
        keep = [int(x) for x in sys.argv[3].split(",") if x.strip()]
        convert(sys.argv[1], sys.argv[2], keep, defs)
        return

    here = os.path.dirname(os.path.abspath(__file__))
    wf = os.path.join(here, "workflows")

    jobs = [
        # 322 Save3DAdvanced <- MeshToFile3D(285) <- MeshSmoothNormals(260)
        #     <- ApplyTextureToMesh(210): the full PBR mesh. That is the deliverable.
        # 282 MeshToFile3D <- PaintMesh(252) is the vertex-coloured mesh -- kept because
        #     it costs nothing extra (same upstream) and is the fallback when UV
        #     unwrapping produces a poor atlas on a thin object like a shelf bracket.
        ("trellis2_image_to_mesh.ui.json", "trellis2_image_to_mesh.api.json", [322, 282]),
    ]
    built = {}
    for src, dst, keep in jobs:
        built[dst] = convert(os.path.join(wf, src), os.path.join(wf, dst), keep, defs)

    # Assert the nodes the handler patches by class actually survived pruning --
    # otherwise the handler would silently fail to inject the input image.
    api = built["trellis2_image_to_mesh.api.json"]
    classes = {n["class_type"] for n in api.values()}
    for required in ("LoadImage", "SaveGLB", "Save3DAdvanced"):
        if required == "SaveGLB":
            continue  # added by the handler, not present in the template
        if required not in classes:
            raise SystemExit(f"expected a {required} node to survive pruning; got {sorted(classes)}")
    ks = [nid for nid, n in api.items() if n["class_type"] == "KSampler"]
    if len(ks) < 3:
        raise SystemExit(f"expected >=3 KSamplers on the TRELLIS.2 path, found {len(ks)}")
    print(f"[build] sanity OK: {len(ks)} KSamplers, LoadImage + Save3DAdvanced present")


if __name__ == "__main__":
    main()
