#!/usr/bin/env bash
# Voice shootout: the SAME script through every model, timed, with the audio kept.
#
#   bash bench.sh <endpoint-id> <workflow.json> <label>
#   bash bench.sh --script                       # print the canonical script and exit
#
# The point is comparability. A benchmark where each model gets prose that happens to suit
# it measures nothing, so the script below is fixed and every lane reads it verbatim -- the
# EN block first, then the FR block. Both are real advertising copy for one of the five
# businesses rather than "the quick brown fox", because the failure modes that matter here
# only appear in real copy: brand names, numbers, units, a street address, and an EN brand
# name sitting inside a FR sentence, which is the single most common way a bilingual read
# falls apart.
#
# Deliberately NOT starting with "Welcome to". VibeVoice hallucinates background music when
# a script opens with a broadcast-style greeting, and a benchmark that triggers a known
# quirk in one lane and not the others is measuring the quirk.
#
# Writes <label>.json (full response), <label>.mp3 (the audio) and appends a row to
# bench-results.tsv. Fill the table in stack-docs/VOICE-BENCHMARK.md from that file.
set -uo pipefail
cd "$(dirname "$0")"
[ -f ../../runpod-stack/.env ] && { set -a; source ../../runpod-stack/.env; set +a; }

read -r -d '' SCRIPT_EN <<'EOF'
Our own crews, our own warehouse, and a quote that does not change after the tear-out.
Engineered white oak, seven and a half inch plank, matte natural oil finish. Fourteen
hundred square feet installed in two days, at eleven dollars and forty cents the square
foot. Come see it at nineteen forty Dundas Street East, or call us and we will bring the
samples to you.
EOF

read -r -d '' SCRIPT_FR <<'EOF'
Nos propres equipes, notre propre entrepot, et un prix qui ne change pas apres la
demolition. Chene blanc d'ingenierie, planche de sept pouces et demi, fini huile mat
naturelle. Quatorze cents pieds carres poses en deux jours, a onze dollars quarante le
pied carre. Venez le voir au dix-neuf cent quarante rue Dundas Est, ou appelez-nous et
nous apporterons les echantillons chez vous.
EOF

if [ "${1:-}" = "--script" ]; then
  printf '=== EN ===\n%s\n\n=== FR ===\n%s\n' "$SCRIPT_EN" "$SCRIPT_FR"
  exit 0
fi

EP="${1:?usage: bench.sh <endpoint-id> <workflow.json> <label>}"
WF="${2:?need a workflow json}"
LABEL="${3:-bench}"
LANG_BLOCK="${BENCH_LANG:-EN}"     # BENCH_LANG=FR bash bench.sh ... to run the French pass

case "$LANG_BLOCK" in
  EN) TEXT="$SCRIPT_EN" ;;
  FR) TEXT="$SCRIPT_FR" ;;
  *)  echo "BENCH_LANG must be EN or FR"; exit 1 ;;
esac

# Splice the script into whichever text-ish field the workflow's TTS node exposes. Node
# packs disagree on the name -- VibeVoice uses "text", Qwen's clone node uses
# "target_text" -- so substitute into every field we know of rather than making each lane
# carry a bespoke workflow that differs in more than the model.
PAYLOAD=$(BENCH_TEXT="$TEXT" python - "$WF" <<'PY'
import json, os, sys
wf = json.load(open(sys.argv[1]))
text = os.environ["BENCH_TEXT"]
wf = {k: v for k, v in wf.items() if not k.startswith("_")}   # strip the _comment doc keys
hit = 0
for node in wf.values():
    ins = node.get("inputs", {})
    for field in ("text", "target_text", "prompt"):
        if field in ins and isinstance(ins[field], str):
            ins[field] = text
            hit += 1
if not hit:
    sys.exit("no text field found in workflow -- benchmark would silently read the default")
print(json.dumps({"input": {"workflow": wf}}))
PY
)
[ -z "$PAYLOAD" ] && { echo "failed to build payload"; exit 1; }

WORDS=$(printf '%s' "$TEXT" | wc -w | tr -d ' ')
START=$(date +%s)
echo "== $LABEL [$LANG_BLOCK, $WORDS words] -> $EP =="

JOB=$(curl -s -X POST "https://api.runpod.ai/v2/$EP/run" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d "$PAYLOAD")
ID=$(printf '%s' "$JOB" | python -c 'import json,sys;print(json.load(sys.stdin).get("id",""))' 2>/dev/null)
[ -z "$ID" ] && { echo "no job id: $JOB"; exit 1; }
echo "job=$ID"

LAST=""
for i in $(seq 1 300); do
  sleep 5
  R=$(curl -s "https://api.runpod.ai/v2/$EP/status/$ID" -H "Authorization: Bearer $RUNPOD_API_KEY")
  S=$(printf '%s' "$R" | python -c 'import json,sys;print(json.load(sys.stdin).get("status",""))' 2>/dev/null)
  [ "$S" != "$LAST" ] && { echo "  [$(( $(date +%s)-START ))s] $S"; LAST="$S"; }
  case "$S" in
    COMPLETED|FAILED|CANCELLED)
      WALL=$(( $(date +%s)-START ))
      printf '%s' "$R" > "$LABEL.json"
      # Decode the audio and measure it. A model that returns COMPLETED with no audio is
      # the failure this whole workstream exists to catch (see patch_audio_outputs.py), so
      # an empty payload is reported loudly rather than scoring as a fast run.
      LABEL="$LABEL" LANG_BLOCK="$LANG_BLOCK" WORDS="$WORDS" WALL="$WALL" \
      printf '%s' "$R" | python - <<'PY'
import base64, json, os, sys, wave, io, contextlib
r = json.load(sys.stdin)
label, lang = os.environ["LABEL"], os.environ["LANG_BLOCK"]
words, wall = int(os.environ["WORDS"]), int(os.environ["WALL"])
out = (r.get("output") or {})
items = out.get("images") or []
if not items:
    print("  NO AUDIO RETURNED. status=%s" % r.get("status"))
    print("  If status is COMPLETED this is the worker-comfyui audio-output bug --")
    print("  the image was built without patch_audio_outputs.py. Rebuild it.")
    print("  errors:", out.get("errors"))
    sys.exit(2)
item = items[0]
fn = item.get("filename", "out.bin")
ext = os.path.splitext(fn)[1] or ".bin"
path = label + ext
if item.get("type") == "base64":
    raw = base64.b64decode(item["data"])
    open(path, "wb").write(raw)
    size = len(raw)
else:
    print("  s3:", item.get("data")); size = -1
dur = None
with contextlib.suppress(Exception):
    with wave.open(path) as w:
        dur = w.getnframes() / w.getframerate()
rtf = (wall / dur) if dur else None
print(f"  wrote {path} ({size/1e6:.2f} MB)" if size > 0 else f"  {path}")
print(f"  wall {wall}s" + (f", audio {dur:.1f}s, RTF {rtf:.3f}" if dur else
                           ", audio duration unknown (non-wav container)"))
hdr = not os.path.exists("bench-results.tsv")
with open("bench-results.tsv", "a") as f:
    if hdr:
        f.write("label\tlang\twords\twall_s\taudio_s\trtf\tbytes\tfile\n")
    f.write(f"{label}\t{lang}\t{words}\t{wall}\t{dur or ''}\t{rtf or ''}\t{size}\t{path}\n")
print("  appended to bench-results.tsv")
PY
      exit 0;;
  esac
done
echo "  timed out waiting"
exit 1
