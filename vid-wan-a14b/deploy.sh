#!/usr/bin/env bash
# Stand up the Wan A14B lane. Run this only AFTER load_weights.sh has printed its sentinel
# and after the image exists in GHCR (CI builds it on a push to main).
#
#   bash deploy.sh [gpu-id]
#
# Differs from runpod-stack/deploy_lane.sh in the one way that matters: the weights are on
# a network volume rather than Model Caching, so the endpoint attaches --network-volume-id
# and is therefore PINNED to EUR-IS-1. That DC is thin -- it listed only A100 SXM 80GB and
# RTX PRO 4500, both Low, when this lane was built.
set -euo pipefail
cd "$(dirname "$0")"
set -a; source ../../runpod-stack/.env; set +a

GPU="${1:-NVIDIA A100-SXM4-80GB}"
VOLUME="yweuz29h2k"
IMAGE="ghcr.io/aspireco/runpod-wan-a14b-worker:latest"
# bf16 on an 80GB card holds both 28.6GB experts resident, so offload is off. On a 48GB
# card pass OFFLOAD=model; see README for the full table.
OFFLOAD="${OFFLOAD:-none}"
DTYPE="${DTYPE:-bf16}"

echo "== lane mos-wan-a14b =="
echo "   gpu     $GPU"
echo "   volume  $VOLUME (pins to EUR-IS-1)"
echo "   dtype   $DTYPE | offload $OFFLOAD"

TPL=$(runpodctl template create \
  --name "mos-wan-a14b" --serverless \
  --image "$IMAGE" \
  --container-disk-in-gb 30 \
  --env "{\"MODEL_DIR\":\"/runpod-volume/wan22-i2v-a14b\",\"OFFLOAD\":\"$OFFLOAD\",\"DTYPE\":\"$DTYPE\"}" \
  | python -c 'import json,sys; print(json.load(sys.stdin).get("id",""))')
[ -z "$TPL" ] && { echo "template creation failed"; exit 1; }
echo "   template $TPL"

# workers-max 1 and idle-timeout 60 are deliberate, not defaults to tune later: idle workers
# bill, and the account caps at 10 max-workers across every endpoint. Park this at 0 the
# moment benchmarking stops.
runpodctl serverless create \
  --name "mos-wan-a14b" \
  --template-id "$TPL" \
  --gpu-id "$GPU" \
  --network-volume-id "$VOLUME" \
  --workers-min 0 --workers-max 1 \
  --idle-timeout 60 \
  --execution-timeout 1800 \
  | python -c '
import json,sys
d=json.load(sys.stdin)
print("   endpoint", d.get("id"), "| max", d.get("workersMax"))
print("   run:", (d.get("urls") or {}).get("run"))
'
echo "== done -- 0 workers until the first request =="
echo "benchmark: python bench.py --lane wan-a14b --endpoint <id> --gpu A100"
echo "park it:   runpodctl serverless update <id> --workers-max 0"
