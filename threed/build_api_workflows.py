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
import re
import sys

sys.path.insert(0, "/comfyui")

WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}


def load_node_defs():
    """Return ComfyUI's NODE_CLASS_MAPPINGS with all extras registered.

    `--cpu` is injected into sys.argv before the import, and it is not optional.
    ComfyUI parses argv in comfy.cli_args at import time, and comfy/model_management.py
    then runs `total_vram = get_total_memory(get_torch_device())` at MODULE level --
    `torch.cuda.current_device()` on a GitHub runner with no GPU, which raises
    "Found no NVIDIA driver" before a single node is registered. The base image's own
    smoke test passes --cpu for exactly this reason.

    Our real arguments were consumed by main() before this runs, so overwriting argv
    here is safe.
    """
    sys.argv = [sys.argv[0], "--cpu"]

    # Setting argv is NOT sufficient on its own, and this is the non-obvious part.
    # comfy/cli_args.py ends with:
    #     if comfy.options.args_parsing: args = parser.parse_args()
    #     else:                          args = parser.parse_args([])
    # and comfy/options.py defaults args_parsing to False. Only main.py flips it, on
    # its first two lines. Import `nodes` directly without flipping it and ComfyUI
    # parses an EMPTY argument list -- so args.cpu is False no matter what argv says,
    # cpu_state stays GPU, and model_management calls torch.cuda.current_device() at
    # module import. That is the "Found no NVIDIA driver" this build hit twice.
    import comfy.options

    comfy.options.enable_args_parsing()

    from comfy.cli_args import args as comfy_args

    if not comfy_args.cpu:
        raise SystemExit(
            "ComfyUI did not accept --cpu (args.cpu is False). Node registration would "
            "initialise CUDA and fail on a GPU-less builder. Check whether "
            "comfy.options.enable_args_parsing() still gates cli_args parsing."
        )

    import nodes

    init = getattr(nodes, "init_extra_nodes", None)
    if init is not None:
        # In ComfyUI 0.34.0 this is `async def init_extra_nodes(init_custom_nodes=True,
        # init_api_nodes=True)`. api_nodes are the paid cloud connectors (Meshy, Rodin,
        # Tripo's hosted API); loading them is pointless here and touches the network
        # during a build, so they are skipped where the signature allows it.
        res = None
        for kwargs in ({"init_custom_nodes": True, "init_api_nodes": False},
                       {"init_custom_nodes": True},
                       {}):
            try:
                res = init(**kwargs)
                break
            except TypeError:
                continue
        if res is not None and hasattr(res, "__await__"):
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

            # Which inputs are sockets is taken from the UI node itself, not guessed
            # from the type name. The editor lists exactly the socket inputs in
            # `inputs`; everything else in the schema is a widget and has a positional
            # entry in `widgets_values`.
            #
            # Guessing from the type is not good enough, and Save3DAdvanced is the
            # proof: its `viewport_state` is a LOAD_3D, which looks like a custom socket
            # type but the frontend renders as a widget and serialises into
            # widgets_values. Treat it as a socket and filename_prefix, width and height
            # all shift by one -- width would receive the empty string. Same class of
            # silent corruption as the control_after_generate slot below.
            sockets = {i["name"] for i in (node.get("inputs") or [])}
            linked = {i["name"]: i["link"] for i in (node.get("inputs") or []) if i.get("link") is not None}
            wv = list(node.get("widgets_values") or [])
            wi = 0
            inputs = {}
            starved = []

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
                if name in sockets:
                    continue  # a socket the template left unconnected
                if wi >= len(wv):
                    # The editor serialises EVERY widget, defaults included, so running
                    # out of values means the schema has an input the editor did not
                    # treat as a widget and did not list as a socket either -- which
                    # means something earlier consumed the wrong slot.
                    starved.append(name)
                    continue
                inputs[name] = wv[wi]
                wi += 1
                # The seed's companion "fixed"/"randomize" slot. Skipping this is the
                # single most damaging bug possible here: every later widget shifts by
                # one and the graph runs anyway, producing wrong output at full cost.
                if opts.get("control_after_generate") and wi < len(wv) and isinstance(wv[wi], str):
                    wi += 1

            # Leftover widget values mean the schema and the serialised list disagree,
            # i.e. something shifted. Fail the BUILD rather than emit a graph whose
            # parameters are quietly wrong.
            if wi != len(wv):
                raise SystemExit(
                    f"{ntype}#{nid}: consumed {wi} of {len(wv)} widget values "
                    f"({wv!r}) -- schema and template disagree, so later inputs would "
                    f"be misaligned. Mapped: {json.dumps(inputs)[:300]}"
                )
            # The mirror-image failure: values ran out while schema inputs remained.
            # Those inputs fall back to node defaults, which may be harmless -- but it
            # also means an earlier non-socket, non-widget input ate a slot meant for
            # something else, and every value after it is on the wrong key. Counting
            # alone cannot tell the two apart, so refuse to guess.
            if starved and wv:
                raise SystemExit(
                    f"{ntype}#{nid}: widget values ran out before inputs {starved} "
                    f"(had {wv!r}). An input is neither a listed socket nor a widget, "
                    "so the positional mapping cannot be trusted."
                )

            api[str(nid)] = {"class_type": ntype, "inputs": inputs, "_meta": {"title": node.get("title", ntype)}}
        return api


SUBGRAPH_TYPE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def convert(ui_path, out_path, keep_ids, node_defs):
    with open(ui_path, encoding="utf-8") as fh:
        ui = json.load(fh)

    # A node whose type is a UUID is a ComfyUI *subgraph* -- a nested graph stored in
    # the workflow's `definitions.subgraphs`, not a registered node class. Expanding one
    # means inlining its nodes and rewriting its boundary links, which this converter
    # does not do. Say so plainly: the alternative is a "node type not installed"
    # message that sends the reader looking for a missing custom node that was never
    # missing. The MoGe scene template is built this way.
    subs = sorted({n["type"] for n in ui.get("nodes", []) if SUBGRAPH_TYPE.match(n.get("type", ""))})
    if subs:
        raise SystemExit(
            f"{os.path.basename(ui_path)} contains {len(subs)} ComfyUI subgraph node(s) "
            f"({subs[0]}...). Subgraph expansion is not implemented; flatten the "
            "template in the ComfyUI editor (right-click the subgraph -> Unpack) and "
            "re-export before converting."
        )

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


def add_save_glb(api, source_ids, prefix="3d/mos"):
    """Terminate the graph with SaveGLB nodes of our own.

    The template ends in Save3DAdvanced, which is deliberately NOT kept: its
    `viewport_state` input is an editor-side widget of a custom type, exactly the shape
    that makes widget alignment ambiguous, and it carries a viewport payload we have no
    use for headless. SaveGLB takes a mesh or a File3D plus a filename prefix and
    nothing else, so there is nothing to misalign.

    MeshToFile3D outputs File3DGLB, which is in SaveGLB's accepted MultiType list.
    """
    added = []
    for i, src in enumerate(source_ids):
        if str(src) not in api:
            raise SystemExit(f"cannot attach SaveGLB: node {src} was pruned")
        nid = str(9001 + i)
        api[nid] = {
            "class_type": "SaveGLB",
            "inputs": {"mesh": [str(src), 0], "filename_prefix": f"{prefix}_{i}"},
            "_meta": {"title": f"mos SaveGLB {i}"},
        }
        added.append(nid)
    return added


def main():
    # Single-template mode, used by the quarantined Hunyuan3D image:
    #   python build_api_workflows.py <in.ui.json> <out.api.json> <keep_id,keep_id,...>
    # Arguments are read BEFORE load_node_defs(), which overwrites sys.argv with --cpu.
    single = sys.argv[1:4] if len(sys.argv) == 4 else None

    defs = load_node_defs()
    print(f"[build] ComfyUI exposes {len(defs)} node types")

    if single:
        keep = [int(x) for x in single[2].split(",") if x.strip()]
        convert(single[0], single[1], keep, defs)
        return

    # Explicit, not derived from __file__: the script is copied to / in the image while
    # the templates live in /workflows, so a path relative to the script resolves to
    # /workflows/workflows and finds nothing.
    wf = os.environ.get("WORKFLOW_DIR", "/workflows")
    if not os.path.isdir(wf):
        raise SystemExit(f"workflow dir {wf!r} does not exist")

    # 285 MeshToFile3D <- MeshSmoothNormals(260) <- ApplyTextureToMesh(210): the full
    #     PBR mesh, and the deliverable.
    # 282 MeshToFile3D <- PaintMesh(252): the vertex-coloured mesh. Kept because it
    #     shares all its upstream with 285 and so costs nothing extra, and it is the
    #     fallback when UV unwrapping produces a poor atlas on a thin object.
    src = os.path.join(wf, "trellis2_image_to_mesh.ui.json")
    dst = os.path.join(wf, "trellis2_image_to_mesh.api.json")
    api = convert(src, dst, [285, 282], defs)

    saves = add_save_glb(api, [285, 282])
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump(api, fh, indent=1)
    print(f"[build] attached SaveGLB nodes {saves}")

    # Assert what the handler relies on. The handler patches by class_type, so a class
    # silently pruned here would mean an input image that is never injected.
    classes = {n["class_type"] for n in api.values()}
    for required in ("LoadImage", "SaveGLB", "ApplyTextureToMesh", "UnwrapMesh"):
        if required not in classes:
            raise SystemExit(f"expected a {required} node to survive pruning; got {sorted(classes)}")
    ks = [nid for nid, n in api.items() if n["class_type"] == "KSampler"]
    if len(ks) < 3:
        raise SystemExit(f"expected >=3 KSamplers on the TRELLIS.2 path, found {len(ks)}")
    print(f"[build] sanity OK: {len(api)} nodes, {len(ks)} KSamplers, LoadImage + SaveGLB present")


if __name__ == "__main__":
    main()
