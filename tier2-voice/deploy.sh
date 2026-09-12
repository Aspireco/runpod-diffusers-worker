#!/usr/bin/env bash
# Stand up the TIER 2 calibration lane. NOTHING THIS ENDPOINT PRODUCES MAY BE PUBLISHED.
#
#   bash runpod-own-worker/tier2-voice/deploy.sh [gpu-id]
#
# The endpoint is named with the t2- prefix ON PURPOSE. pipeline/tier_guard.py refuses that
# prefix before it even consults the registry, so a Tier 2 endpoint reached by raw id or by
# name -- including one nobody remembered to register -- still cannot enter the publishing
# path. The name is load-bearing: do not "tidy" it.
#
# No --model-reference: the weights are baked into the image (XTTS v2 is ~1.8GB and the TTS
# library wants its own on-disk layout, not an HF snapshot tree). That also means the CPML
# licence text ships inside the image beside the checkpoint it governs.
#
# COST. workers-min 0 and a 60s idle timeout, like every lane here. Idle workers bill; this
# one is a benchmark, so PARK IT AT max=0 the moment the comparison run finishes:
#
#   runpodctl serverless update <id> --workers-max 0
#
# The account caps at 10 max-workers across all endpoints and most slots are in use, so this
# lane is borrowing a slot, not claiming one.
set -euo pipefail
cd "$(dirname "$0")"
REPO_ROOT="$(cd ../.. && pwd)"
set -a; source "$REPO_ROOT/runpod-stack/.env"; set +a

# Match the Tier 1 lane's card, because generation times measured across different GPU
# classes are not comparable and speed is half of what this benchmark is for.
#
# That card is a 4090 (ADA_24), NOT the AMPERE_48 the production mos-tts endpoint uses.
# AMPERE_48 had no capacity on 2026-09-12 -- a worker sat `throttled` for 15 minutes and the
# first benchmark job timed out -- so the Tier 1 comparison point was rebuilt from the same
# template (n8liwbp6dc) onto ADA_24 as endpoint est1ckabyygzc4. Both tiers on ADA_24 keeps
# the comparison valid; XTTS needs ~4GB of VRAM so 24GB is not a constraint.
GPU="${1:-NVIDIA GeForce RTX 4090}"
IMAGE="ghcr.io/aspireco/runpod-tier2-voice-worker:latest"
NAME="t2-xtts-v2"

echo "== TIER 2 CALIBRATION LANE =="
echo "   model   coqui/XTTS-v2"
echo "   licence Coqui Public Model License -- NON-COMMERCIAL, and Coqui is defunct so no"
echo "           commercial licence can be bought from anyone. Output is never published."
echo "   gpu     $GPU"
echo ""

TPL=$(runpodctl template create \
  --name "$NAME" --serverless \
  --image "$IMAGE" \
  --container-disk-in-gb 25 \
  --env "{\"TIER2_ENDPOINT_NAME\":\"$NAME\"}" \
  | python -c 'import json,sys; print(json.load(sys.stdin).get("id",""))')

[ -z "$TPL" ] && { echo "template creation failed"; exit 1; }
echo "   template $TPL"

runpodctl serverless create \
  --name "$NAME" \
  --template-id "$TPL" \
  --gpu-id "$GPU" \
  --workers-min 0 --workers-max 1 \
  --idle-timeout 60 \
  --execution-timeout 600 \
  | python -c '
import json,sys
d = json.load(sys.stdin)
eid = d.get("id")
print("   endpoint", eid, "| gpu", d.get("gpuIds"), "| max", d.get("workersMax"))
print("")
print("   Record this id under tier2.t2-xtts-v2.endpoint_id in")
print("   models/testing/TIER_REGISTRY.json, then run the comparison:")
print("")
print("     python models/testing/compare.py voice \\")
print("        --tier1 kqow92p29gl9bn --tier1-label chatterbox \\")
print("        --tier1-model ResembleAI/chatterbox --tier1-license mit \\")
print("        --tier2", eid, "--tier2-label t2-xtts-v2 \\")
print("        --tier2-model coqui/XTTS-v2 --tier2-license coqui-public-model-license")
print("")
print("   THEN PARK IT:  runpodctl serverless update", eid, "--workers-max 0")
'
echo "== done -- 0 workers until the first request =="
