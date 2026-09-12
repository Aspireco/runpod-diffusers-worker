#!/usr/bin/env bash
# Fill the 150GB EUR-IS-1 volume with Wan2.2-I2V-A14B-Diffusers (126.2GB, 41 files).
#
#   bash load_weights.sh
#
# Descended from runpod-stack/load_weights.sh and keeps its four hard-won rules:
#   1. utility pods do NOT use runpod/pytorch:* -- several DCs cannot reach
#      registry-1.docker.io and the pod burns money retrying with no logs
#   2. HF_HOME/HF_HUB_CACHE point at the volume, never the 20GB container disk
#   3. a unique sentinel only our own script can emit marks completion
#   4. autokill armed the moment the pod exists
#
# Three more rules were paid for by this script's own first two runs:
#
#   5. The payload is base64'd rather than interpolated into --docker-args. The LTX version
#      embedded a shell script inside a double-quoted string inside JSON; one stray quote
#      in a later edit silently truncates the command. base64 has no metacharacters.
#
#   6. `hf download --max-workers 4` on a Runpod CPU pod is OOM-killed, every time. CPU pods
#      default to 2 vCPU / 4GB RAM, and four concurrent ~5GB shards landing on a NETWORK
#      volume hold their dirty pages in RAM until writeback drains. In the logs this looks
#      exactly like a network failure -- "Killed" after a truncated progress bar -- which is
#      why the retry loop misreported it as a 429.
#
#   7. `hf download` calls https://huggingface.co/api/models/<repo>/revision/main once per
#      invocation, and THAT endpoint is what returns HTTP 429 after a day of large pulls.
#      The file bytes live on a different host (cdn-lfs) that was never throttled. So this
#      does not use `hf download` at all: the file list is a checked-in manifest and each
#      file is fetched straight from /resolve/main/<path> with curl. One throttled metadata
#      call is removed from the critical path, and a per-file size check replaces trusting
#      any exit code.
set -uo pipefail
cd "$(dirname "$0")"
set -a; source ../../runpod-stack/.env; set +a

REPO="Wan-AI/Wan2.2-I2V-A14B-Diffusers"
DEST="/workspace/wan22-i2v-a14b"
VOLUME="yweuz29h2k"          # mos-video-weights, 150GB, EUR-IS-1
DC="EUR-IS-1"
KILL_AFTER_MIN=180
SENTINEL="WAN_A14B_WEIGHTS_READY_7f3c91"

[ -s manifest.tsv ] || { echo "manifest.tsv missing -- regenerate it first"; exit 1; }
MANIFEST=$(cat manifest.tsv)
NEED=$(awk -F'\t' '{s+=$2} END {print s}' manifest.tsv)
echo "manifest: $(wc -l < manifest.tsv) files, $(echo "$NEED" | awk '{printf "%.1f", $1/1e9}') GB"

read -r -d '' DL <<SCRIPT
set -u
export HF_HOME=/workspace/.hf
export HF_HUB_CACHE=/workspace/.hf/hub
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p /workspace/.hf/hub $DEST
cd $DEST

cat > /tmp/manifest.tsv <<'MANIFEST_EOF'
$MANIFEST
MANIFEST_EOF

echo "[dl] \$(wc -l < /tmp/manifest.tsv) files to fetch"

# Per file, up to 6 attempts. curl -C - resumes from whatever is already on the volume, so
# a partial file from an earlier pod is progress rather than garbage, and a file already at
# its manifest size costs one stat() instead of a network round trip. --retry-all-errors
# covers the transient 5xx that a CDN throws under load without ending the attempt.
FAILED=0
while IFS=\$'\t' read -r path size; do
  [ -z "\$path" ] && continue
  mkdir -p "\$(dirname "\$path")"
  have=\$(stat -c %s "\$path" 2>/dev/null || echo 0)
  if [ "\$have" = "\$size" ]; then
    echo "[skip] \$path (\$size bytes already present)"
    continue
  fi
  ok=0
  for attempt in 1 2 3 4 5 6; do
    echo "[get ] \$path attempt \$attempt (have \$have of \$size)"
    curl -fSL --retry 5 --retry-delay 10 --retry-all-errors --connect-timeout 30 \
         -H "Authorization: Bearer \$HF_TOKEN" \
         -C - -o "\$path" \
         "https://huggingface.co/$REPO/resolve/main/\$path"
    rc=\$?
    # Writeback on a network volume lags the transfer; sync before measuring, or a good
    # file looks short and the loop re-downloads 5GB for nothing.
    sync
    have=\$(stat -c %s "\$path" 2>/dev/null || echo 0)
    if [ "\$have" = "\$size" ]; then ok=1; break; fi
    # curl exit 33 means the server refused a range request and the local file is already
    # whole or unusable; 22 is an HTTP >=400, which for us means the 429 arrived here too.
    echo "[warn] \$path rc=\$rc have=\$have want=\$size; backing off"
    sleep \$(( attempt * 45 ))
  done
  if [ "\$ok" != "1" ]; then echo "[FAIL] \$path"; FAILED=\$((FAILED+1)); fi
done < /tmp/manifest.tsv

echo "[dl] finished pass, \$FAILED file(s) failed"

# Trust bytes on disk, never an exit code. The sentinel is emitted only by this check.
python3 - <<'PYEOF'
import os, sys
DEST = "$DEST"
bad, total = [], 0
for line in open("/tmp/manifest.tsv"):
    line = line.rstrip("\n")
    if not line:
        continue
    path, size = line.split("\t")
    size = int(size)
    full = os.path.join(DEST, path)
    have = os.path.getsize(full) if os.path.exists(full) else -1
    total += max(have, 0)
    if have != size:
        bad.append((path, have, size))
print(f"[verify] {total/1e9:.1f} GB on disk across the manifest")
# The two MoE experts are the whole point of the A14B. A download that quietly dropped
# transformer_2 would look nearly complete and produce mush below the MoE boundary.
for sub in ("transformer", "transformer_2", "text_encoder", "vae"):
    d = os.path.join(DEST, sub)
    print(f"[verify] {sub}: {len(os.listdir(d)) if os.path.isdir(d) else 0} files")
if bad:
    print(f"[verify] FAIL: {len(bad)} file(s) wrong size")
    for p, h, s in bad[:10]:
        print(f"   {p}: have {h} want {s}")
    sys.exit(1)
print("[verify] OK: every manifest file matches its exact byte size")
PYEOF

if [ \$? -eq 0 ]; then
  echo "$SENTINEL" > /workspace/WAN_A14B_DONE.txt
  du -sb $DEST >> /workspace/WAN_A14B_DONE.txt
  echo "$SENTINEL"
else
  echo "WAN_A14B_INCOMPLETE"
fi
sleep infinity
SCRIPT

B64=$(printf '%s' "$DL" | base64 -w0)
CMD="bash -c \"echo $B64 | base64 -d | bash\""

# curl streams one file at a time straight to disk, with an explicit sync after each, so
# the 4GB CPU pod that OOM-killed `hf download` is now the right instance: it is $0.06/hr
# against $1.39 for the A100 this otherwise falls back to. GPU pods are kept as a fallback
# purely for their RAM, and ordered by what EUR-IS-1 actually stocks -- the DC is thin, and
# a generic cheapest-first list once walked six GPU types and got "none" from every one.
POD=""
try_create() {
  OUT=$(runpodctl pod create --name "mos-wan-a14b-loader" "$@" \
    --image "ghcr.io/aspireco/runpod-stt-worker:latest" \
    --network-volume-id "$VOLUME" --data-center-ids "$DC" \
    --container-disk-in-gb 20 \
    --env "{\"HF_TOKEN\":\"$HF_TOKEN\"}" \
    --docker-args "$CMD" 2>&1)
  printf '%s' "$OUT" | python -c 'import json,sys
try:
    d=json.load(sys.stdin); print("" if "error" in d else d.get("id",""))
except Exception: print("")' 2>/dev/null
}

printf '%-46s %-10s ' "CPU (2vcpu/4gb)" "SECURE"
POD=$(try_create --compute-type cpu)
if [ -n "$POD" ]; then echo "ALLOCATED $POD"; else
  echo "none"
  for g in "NVIDIA RTX PRO 4500" "NVIDIA A100-SXM4-80GB" "NVIDIA RTX A6000" "NVIDIA L40S"; do
    for ct in SECURE COMMUNITY; do
      printf '%-46s %-10s ' "$g" "$ct"
      POD=$(try_create --gpu-id "$g" --cloud-type "$ct")
      if [ -n "$POD" ]; then echo "ALLOCATED $POD"; break 2; fi
      echo "none"
    done
  done
fi

if [ -z "$POD" ]; then
  echo "Nothing allocatable in $DC right now. Volume idles at \$10.50/month; retry later."
  exit 1
fi

echo "$POD" > .loader_pod
echo "autokill: $POD deleted in $KILL_AFTER_MIN min regardless of outcome"
nohup bash ../../runpod-stack/autokill.sh "$POD" "$KILL_AFTER_MIN" >autokill.log 2>&1 &
echo "pod $POD pulling 126.2GB ($REPO) -> $DEST"
echo "watch:  runpodctl pod logs $POD | tail"
echo "done when the log prints: $SENTINEL"
