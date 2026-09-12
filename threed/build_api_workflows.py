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

WHAT MAKES THE POSITIONAL MAPPING HARD
--------------------------------------
Almost all of the difficulty is one question: which schema inputs occupy a slot in
`widgets_values`, and in what order. Getting it wrong does not raise -- it assigns
values to the wrong keys and the graph runs anyway, producing garbage at full GPU
cost. Every rule below was learned from a node in this template that broke a simpler
one, which is why to_api() also ASSERTS that the slots and the values balance.

1. Companion widgets. Some inputs serialise TWO values. A seed emits its value then
   its control mode ("fixed"), and LoadImage's upload button emits the filename then
   its type ("image") -- one input, two slots, in both cases. See
   COMPANION_WIDGET_OPTS.

2. A converted widget keeps its slot. UnwrapMesh's `resolution` is an INT wired from
   a PrimitiveInt: it is a socket AND it still holds its position in widgets_values.
   Skip it and `padding` gets 2048.

3. A socket-looking type can be a widget. Save3DAdvanced's `viewport_state` is a
   LOAD_3D -- not a widget type, not listed among the node's sockets, and it holds a
   slot. Skip it and `width` receives the empty string.

4. Declaration order, not bucketed order. V3 nodes declare one ordered input list;
   the V1 INPUT_TYPES view splits it into required/optional and loses the
   interleaving. define_schema() is asked first.

Two structural fixups on top of that: output nodes used as sources (the template taps
`ApplyTextureToMesh.base_color` off a PreviewImage, which has no output slot in any
ComfyUI version) are resolved through to the real source, and muted/bypassed nodes are
reached through rather than emitted. Finally the graph is pruned to the ancestors of
the requested outputs -- dropping five unused preview renders and a UV-atlas render,
the difference between paying for one texture bake and six.
"""

import json
import os
import re
import sys

sys.path.insert(0, "/comfyui")

WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}

# Input options whose presence means the editor renders a SECOND widget for the same
# input, and therefore serialises an extra positional value that the schema does not
# mention. Discovered the hard way, one build at a time; the leftover-value assertion
# in to_api() is what surfaces a new one rather than letting it corrupt the mapping.
# Which of the two pipelines the vendor template carries. See select_pipeline().
USE_TRELLIS2 = True

COMPANION_WIDGET_OPTS = ("control_after_generate", "image_upload", "video_upload",
                         "audio_upload", "animated_image_upload", "model_upload")


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
    """Ordered [(name, type, options, dynamic_options)] for a node class.

    Order is load-bearing: widget values are matched to inputs positionally, so getting
    the sequence wrong silently assigns values to the wrong keys.

    INPUT_TYPES() alone is not a reliable source of it. V3 nodes declare a single
    ordered `inputs` list mixing required and optional entries, and the V1 view buckets
    them into {"required": ..., "optional": ...} -- order survives inside each bucket
    but the interleaving is lost. So define_schema() supplies the order and INPUT_TYPES
    only the types.

    The fourth element is for DynamicCombo inputs. Those advertise every sub-input of
    EVERY option in the flattened V1 view (DynamicCombo.Input.get_dynamic() returns
    `[self] + [i for option in self.options for i in option.inputs]`), while the editor
    serialises only the sub-widgets of the SELECTED option. RemeshMesh is the example:
    `sign_mode` picks between a "udf" branch with three booleans and an "sdf" branch
    with two, so its 9 schema inputs serialise as 11 widget values when "udf" is
    chosen. Which extra slots exist cannot be known until the combo's value is read,
    so the option objects are carried through to the caller.
    """
    it = cls.INPUT_TYPES()
    meta = {}
    for section in ("required", "optional"):
        for name, spec in (it.get(section) or {}).items():
            if isinstance(spec, (list, tuple)) and spec:
                typ = spec[0]
                opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            else:
                typ, opts = spec, {}
            meta[name] = (typ, opts)

    order, dynamic = None, {}
    define = getattr(cls, "define_schema", None)
    if define is not None:
        try:
            schema_inputs = define().inputs or []
            names = []
            for i in schema_inputs:
                nid = getattr(i, "id", None) or getattr(i, "name", None)
                if nid is None:
                    continue
                names.append(nid)
                opts_list = getattr(i, "options", None)
                # A DynamicCombo's options are Option objects carrying .key/.inputs.
                # Plain combos also have list-ish options, so require the shape.
                if opts_list and all(hasattr(o, "key") and hasattr(o, "inputs") for o in opts_list):
                    dynamic[nid] = list(opts_list)
            # Names the schema does not list are the flattened dynamic sub-inputs; they
            # are emitted inline when their branch is selected, so they are dropped here
            # rather than being treated as top-level positional widgets.
            if names and len(set(names)) == len(names) and all(n in meta for n in names):
                order = names
        except Exception:
            order, dynamic = None, {}

    if order is None:
        order = list(meta)
    return [(n, meta[n][0], meta[n][1], dynamic.get(n)) for n in order]


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

    def switch_value(self, node):
        """The boolean an If/Else switch is set to: a literal widget, or a primitive.

        The template drives three switches from one PrimitiveBoolean, so `switch` is a
        LINKED input on those and its widgets_values entry is the stale placeholder a
        converted widget leaves behind. Read the primitive in that case.
        """
        sw = next((i for i in (node.get("inputs") or []) if i.get("name") == "switch"), None)
        if sw is not None and sw.get("link") is not None:
            origin = self.links.get(sw["link"])
            if origin:
                src = self.nodes.get(origin[0])
                wv = (src or {}).get("widgets_values") or []
                if wv:
                    return bool(wv[0])
            return None
        wv = node.get("widgets_values") or []
        return bool(wv[0]) if wv else None

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

        # An If/Else switch is resolved HERE, at build time, rather than left in the
        # graph. Its branches are declared lazy=True, so ComfyUI would only execute the
        # selected one -- but /prompt still VALIDATES the whole graph, so an unselected
        # UNETLoader whose checkpoint is absent gets the prompt rejected outright.
        # Resolving the switch prunes the dead branch along with everything only it
        # feeds, which is what keeps a second 5GB checkpoint out of the image.
        if node["type"] == "ComfySwitchNode":
            val = self.switch_value(node)
            if val is None:
                raise SystemExit(
                    f"ComfySwitchNode#{nid}: cannot determine its boolean at build time; "
                    "it is neither a literal widget nor driven by a primitive"
                )
            want = "on_true" if val else "on_false"
            tgt = next((i for i in (node.get("inputs") or []) if i.get("name") == want), None)
            if tgt is None or tgt.get("link") is None:
                raise SystemExit(f"ComfySwitchNode#{nid}: selected branch {want!r} is not connected")
            return self.resolve(tgt["link"], _seen)

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
        mismatches = []
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

            # DOES THIS INPUT OCCUPY A POSITIONAL SLOT IN widgets_values?
            #
            # Neither "is it a widget type" nor "is it listed as a socket" answers this
            # alone. Both real nodes in this template disprove the simple rules:
            #
            #   UnwrapMesh   `resolution` is an INT wired from a PrimitiveInt. It is a
            #                socket (it appears in `inputs`, with a link) AND it still
            #                holds its slot in widgets_values -- the editor keeps the
            #                placeholder for a widget that was converted to an input.
            #                Skip it and `padding` receives 2048 and `weld_distance`
            #                receives 1.
            #   Save3DAdvanced  `viewport_state` is a LOAD_3D. Not a widget type by any
            #                reasonable reading, not listed among the sockets, and it
            #                occupies a slot. Skip it and `width` receives "".
            #
            # So: a slot is held by anything the editor could render as a widget (a
            # widget-typed input, even when currently linked) OR anything the schema
            # declares that the editor did not list as a socket. A link, when present,
            # then overrides the value -- but the slot is consumed either way.
            sockets = {i["name"] for i in (node.get("inputs") or [])}
            linked = {i["name"]: i["link"] for i in (node.get("inputs") or []) if i.get("link") is not None}
            wv = list(node.get("widgets_values") or [])
            wi = 0
            inputs = {}
            starved = []

            for name, typ, opts, dyn_options in input_spec(cls):
                holds_slot = is_widget(typ, opts) or name not in sockets
                value, got = None, False
                if holds_slot:
                    if wi < len(wv):
                        value, got = wv[wi], True
                        wi += 1
                        # Some inputs carry a COMPANION widget that the editor
                        # serialises as an extra positional value the schema never
                        # mentions:
                        #   control_after_generate  a seed's "fixed"/"randomize" mode
                        #   image_upload            LoadImage's upload button, which
                        #                           emits its type ("image") after the
                        #                           filename -- two values for one input
                        # Miss one and every later widget shifts by one, and the graph
                        # still runs, producing wrong output at full GPU cost with no
                        # error. The guard below is only a cross-check; this is the fix.
                        if (any(opts.get(k) for k in COMPANION_WIDGET_OPTS)
                                and wi < len(wv) and isinstance(wv[wi], str)):
                            wi += 1

                        # A DynamicCombo's SELECTED branch contributes one slot per
                        # sub-input, immediately after the combo's own value. They are
                        # emitted as ordinary named inputs, which is how the editor
                        # posts them; ComfyUI's dynamic-input layer reassembles them
                        # into the dict execute() receives.
                        if dyn_options:
                            branch = next((o for o in dyn_options
                                           if str(getattr(o, "key", "")) == str(value)), None)
                            if branch is None:
                                mismatches.append(
                                    f"{ntype}#{nid}.{name}: value {value!r} matches no "
                                    f"DynamicCombo option ({[getattr(o, 'key', '?') for o in dyn_options]})"
                                )
                            else:
                                for sub in getattr(branch, "inputs", []):
                                    sid = getattr(sub, "id", None) or getattr(sub, "name", None)
                                    if sid is None:
                                        continue
                                    if wi < len(wv):
                                        inputs[sid] = wv[wi]
                                        wi += 1
                                    else:
                                        starved.append(sid)
                    else:
                        starved.append(name)

                if name in linked:
                    src = self.resolve(linked[name])
                    if src is None:
                        raise SystemExit(
                            f"{ntype}#{nid}.{name}: link {linked[name]} dead-ends "
                            "(muted node with nothing upstream?)"
                        )
                    inputs[name] = [str(src[0]), src[1]]
                elif got:
                    inputs[name] = value

            # Slots and values must balance. Leftover values mean the schema and the
            # serialised list disagree; values running out means an input is neither a
            # listed socket nor a recognised widget and something earlier ate the wrong
            # slot. Either way the positional mapping cannot be trusted.
            #
            # These are COLLECTED, not raised on the spot. Each discovery costs a full
            # image build, and failing on the first bad node hides the rest -- three
            # separate builds were spent learning three separate node shapes one at a
            # time before this was changed to report them all at once.
            if wi != len(wv):
                mismatches.append(
                    f"{ntype}#{nid}: consumed {wi} of {len(wv)} widget values {wv!r}; "
                    f"mapped {json.dumps(inputs)[:200]}"
                )
            elif starved and wv:
                mismatches.append(
                    f"{ntype}#{nid}: values ran out before inputs {starved} (had {wv!r})"
                )

            api[str(nid)] = {"class_type": ntype, "inputs": inputs, "_meta": {"title": node.get("title", ntype)}}

        if mismatches:
            detail = "\n  - ".join(mismatches)
            raise SystemExit(
                f"{len(mismatches)} node(s) whose widget slots and values do not "
                f"balance -- the positional mapping would be wrong:\n  - {detail}\n\n"
                "Usually one more input carries a companion widget; add its option "
                "key to COMPANION_WIDGET_OPTS."
            )
        return api


SUBGRAPH_TYPE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def select_pipeline(ui, use_trellis2=True):
    """Set the template's pipeline switch before conversion.

    THIS IS NOT COSMETIC. ComfyUI ships this workflow as "Pixal3D & TRELLIS.2: Image to
    Model", and the PrimitiveBoolean driving its three If/Else switches -- titled
    "Boolean (Switch to Trellis2)" -- ships set to **False**. Straight out of the box the
    template runs the PIXAL3D pipeline, loading pixal3d_int8_convrot.safetensors, not
    TRELLIS.2 at all.

    Both are MIT (Pixal3D is TencentARC's), so either is usable. TRELLIS.2 is the
    documented pick for this lane, so the choice is made explicitly here rather than
    inherited from whatever the vendor happened to save. Flipping this to False and
    downloading the Pixal3D checkpoint is the whole of switching pipelines.
    """
    nodes = {n["id"]: n for n in ui.get("nodes", [])}
    links = {}
    for l in ui.get("links") or []:
        links[l["id"] if isinstance(l, dict) else l[0]] = (
            (l["origin_id"], l["origin_slot"]) if isinstance(l, dict) else (l[1], l[2]))

    driven = set()
    for n in ui.get("nodes", []):
        if n.get("type") != "ComfySwitchNode":
            continue
        sw = next((i for i in (n.get("inputs") or []) if i.get("name") == "switch"), None)
        if sw and sw.get("link") is not None:
            origin = links.get(sw["link"])
            if origin:
                driven.add(origin[0])

    if not driven:
        print("[build] no primitive-driven pipeline switch found; template default stands")
        return
    for nid in sorted(driven):
        node = nodes.get(nid)
        if node is None:
            continue
        before = (node.get("widgets_values") or [None])[0]
        node["widgets_values"] = [bool(use_trellis2)]
        print(f"[build] pipeline switch #{nid} ({node.get('title', '')!r}): "
              f"{before} -> {bool(use_trellis2)} "
              f"({'TRELLIS.2' if use_trellis2 else 'Pixal3D'})")


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

    select_pipeline(ui, use_trellis2=USE_TRELLIS2)

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


MODEL_SUFFIXES = (".safetensors", ".ckpt", ".pth", ".pt", ".bin", ".onnx")


def verify_models(api, models_root="/comfyui/models"):
    """Assert every model filename the graph asks for is actually on disk.

    ComfyUI's `--quick-test-for-ci` boot registers nodes but does not load weights, so
    a loader pointing at a filename that was never downloaded passes every other check
    in this build and fails on a GPU that bills by the second.

    This is not hypothetical here. The template's CLIPVisionLoader wants
    `dino_v3_L_naf_fp32.safetensors`, which lives in Comfy-Org/Pixal3D -- while
    Comfy-Org/TRELLIS.2, the obvious place to look, ships a differently-named
    `dino_v3_vit_l.safetensors`. Downloading the wrong one is a one-word mistake that
    this check turns into a build failure.
    """
    if not os.path.isdir(models_root):
        print(f"[build] {models_root} absent, skipping model presence check")
        return

    on_disk = {}
    for dirpath, _dirs, files in os.walk(models_root):
        for f in files:
            on_disk.setdefault(f, os.path.join(dirpath, f))

    wanted, missing = set(), []
    for nid, node in api.items():
        for name, val in node["inputs"].items():
            if isinstance(val, str) and val.lower().endswith(MODEL_SUFFIXES):
                wanted.add(val)
                if val not in on_disk:
                    missing.append(f"{node['class_type']}#{nid}.{name} -> {val}")

    if missing:
        raise SystemExit(
            "the graph references model files that are not in the image:\n  - "
            + "\n  - ".join(missing)
            + f"\npresent under {models_root}:\n  "
            + "\n  ".join(sorted(on_disk)) or "(none)"
        )
    print(f"[build] all {len(wanted)} referenced model files present: {sorted(wanted)}")


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
    # The template ships set to Pixal3D, so a refreshed template that silently flips
    # back would otherwise be caught only by a 5GB download failing. Name it here.
    other = sorted({v for n in api.values() for v in n["inputs"].values()
                    if isinstance(v, str) and "pixal3d" in v.lower()})
    if USE_TRELLIS2 and other:
        raise SystemExit(
            f"pipeline is set to TRELLIS.2 but the graph still references {other} -- "
            "the switch resolution did not prune the Pixal3D branch"
        )

    ks = [nid for nid, n in api.items() if n["class_type"] == "KSampler"]
    if len(ks) < 3:
        raise SystemExit(f"expected >=3 KSamplers on the TRELLIS.2 path, found {len(ks)}")
    print(f"[build] sanity OK: {len(api)} nodes, {len(ks)} KSamplers, LoadImage + SaveGLB present")

    verify_models(api)


if __name__ == "__main__":
    main()
