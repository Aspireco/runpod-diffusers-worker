#!/bin/sh
# fetch_model.sh <url> <dest-dir> <filename> <min-bytes>
#
# WHY THIS EXISTS. `comfy model download` retries three times and gives up. This image
# pulls ~9.6GB from HuggingFace, and after a few rebuilds in an hour HF starts
# answering 429 Too Many Requests -- which killed a build on the MoGe weights after the
# 5.25GB TRELLIS.2 pull had already succeeded. The project has hit this before (see
# DECISIONS.md 0.3, where four LTX-2.5 attempts in an hour earned a rate-limit that
# needed hours of backoff).
#
# So: long backoff that actually waits out a 429, resume support so a dropped 5GB
# transfer does not restart from zero, and a SIZE ASSERTION -- because the failure that
# matters is not a download that errors, it is one that writes an HTML error page to
# a .safetensors path and exits 0. That would sail through the build and fail at model
# load time on a billing GPU.
set -e

URL="$1"; DIR="$2"; NAME="$3"; MIN="$4"
OUT="$DIR/$NAME"

mkdir -p "$DIR"

if [ -s "$OUT" ]; then
  SZ=$(stat -c %s "$OUT")
  if [ "$SZ" -ge "$MIN" ]; then
    echo "[fetch] $NAME already present ($SZ bytes)"
    exit 0
  fi
  echo "[fetch] $NAME present but undersized ($SZ < $MIN), refetching"
  rm -f "$OUT"
fi

# --retry-all-errors so a 429/5xx is retried rather than treated as final; -C - to
# resume a partial transfer instead of re-pulling gigabytes.
#
# Arguments are built with `set --` rather than an inline ${HF_TOKEN:+...}: in POSIX sh
# the quotes inside that expansion are literal, so a token would be sent as a header
# named `Authorization:` with the rest split on spaces. Every repo here is public, so
# the token is optional -- but a silently malformed header is worse than none, and this
# is the shape that bites the day someone adds a gated model.
set -- -fL --progress-bar \
       --retry 10 --retry-all-errors --retry-delay 20 --retry-max-time 2400 \
       --connect-timeout 30 -C -
if [ -n "${HF_TOKEN:-}" ]; then
  set -- "$@" -H "Authorization: Bearer $HF_TOKEN"
fi

curl "$@" -o "$OUT.part" "$URL"

mv "$OUT.part" "$OUT"

SZ=$(stat -c %s "$OUT")
if [ "$SZ" -lt "$MIN" ]; then
  echo "[fetch] FAILED: $NAME is $SZ bytes, expected at least $MIN."
  echo "[fetch] First bytes (an HTML error page written to a .safetensors path is the"
  echo "[fetch] failure this check exists to catch):"
  head -c 200 "$OUT" || true
  exit 1
fi

# Structural check, on top of the size check. A safetensors file opens with a
# little-endian u64 giving the JSON header's length; that header is never 4GB, so bytes
# 4..7 are always zero. Text -- HTML, JSON, a git-lfs pointer, a Markdown error page --
# essentially never has four NUL bytes there.
#
# Sniffing the first character for '<' or '{' is NOT enough: a plain-text error body can
# begin with anything. This was caught by testing the guard against a real README, which
# starts with '-' and sailed straight through the earlier version of this check.
case "$NAME" in
  *.safetensors)
    HEAD_HI=$(od -An -tx1 -j4 -N4 "$OUT" | tr -d ' \n')
    if [ "$HEAD_HI" != "00000000" ]; then
      echo "[fetch] FAILED: $NAME has no safetensors header (bytes 4-7 = $HEAD_HI, expected 00000000)."
      echo "[fetch] That means the body is not a tensor file. First bytes:"
      head -c 200 "$OUT" || true
      exit 1
    fi
    ;;
esac

echo "[fetch] OK $OUT ($SZ bytes)"
