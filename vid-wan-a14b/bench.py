"""Same prompt, every video lane, one table.

    python bench.py --lane wan-a14b --endpoint <id> [--image <path-or-url>]
    python bench.py --replay                       # rebuild the table from saved runs

The point of this script is that a quality claim is only worth something if every model
answered the *same* question. The prompt, the seed, the frame count and the source image
are fixed here rather than passed in, so a lane cannot flatter itself with a friendlier
brief. Per-lane sampling settings (steps, guidance) are allowed to differ because a
distilled 4-step model run at 40 steps is not "the same test", it is a broken one.

Results land in results/<lane>.json plus the mp4, and --replay prints the markdown table
that goes into stack-docs/VIDEO-BENCHMARK.md. Cost is computed from measured wall-clock
against the card's posted hourly rate, because that is the number that decides anything.
"""

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

# The standard brief. Deliberately a product-marketing shot rather than a cinematic set
# piece: this stack exists to sell flooring, not to win a film festival. It asks for the
# three things that separate tiers in practice -- a slow believable camera move, a real
# material that betrays texture errors, and stable lighting that exposes flicker.
PROMPT = (
    "Slow cinematic dolly push-in across a wide-plank white oak floor in a sunlit modern "
    "living room, warm morning light raking from tall windows, fine wood grain and subtle "
    "sheen visible, dust motes drifting in the light, shallow depth of field, "
    "photorealistic, steady smooth camera motion"
)
NEGATIVE = (
    "static, still image, low quality, blurry, watermark, text, distorted, warped, "
    "flickering, jpeg artifacts, oversaturated, cartoon"
)
SEED = 1234
NUM_FRAMES = 81          # 5.06s at 16fps; lanes at other fps get the nearest 4n+1
WIDTH, HEIGHT = 1280, 720

# Posted Runpod on-demand rates, USD/hr. Cost per clip is wall-clock * rate, which counts
# cold start -- that is honest, because a lane that scales to zero pays it on every burst.
GPU_RATES = {
    "A6000": 0.53, "A40": 0.53, "4090": 0.69, "L40S": 0.99,
    "A100": 1.39, "RTXPRO6000": 1.99, "RTXPRO4500": 0.34, "A5000": 0.26,
}


def call(endpoint: str, payload: dict, api_key: str, timeout: int = 2400) -> dict:
    """Submit async and poll. /runsync drops the connection well before a 720p video
    finishes on an offloaded card, and the job keeps billing after the client gives up."""
    base = f"https://api.runpod.ai/v2/{endpoint}"
    hdr = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    req = urllib.request.Request(f"{base}/run",
                                 data=json.dumps(payload).encode(), headers=hdr)
    with urllib.request.urlopen(req, timeout=120) as r:
        job = json.load(r)
    jid = job.get("id")
    if not jid:
        return {"error": f"no job id: {job}"}
    print(f"  job {jid} queued", flush=True)

    t0 = time.time()
    last = ""
    while time.time() - t0 < timeout:
        time.sleep(5)
        try:
            sreq = urllib.request.Request(f"{base}/status/{jid}", headers=hdr)
            with urllib.request.urlopen(sreq, timeout=60) as r:
                st = json.load(r)
        except urllib.error.URLError as e:
            print(f"  poll error {e}; retrying", flush=True)
            continue
        s = st.get("status")
        if s != last:
            print(f"  [{int(time.time() - t0):>4}s] {s}", flush=True)
            last = s
        if s == "COMPLETED":
            st["_wall_seconds"] = round(time.time() - t0, 1)
            return st
        if s in ("FAILED", "CANCELLED", "TIMED_OUT"):
            st["_wall_seconds"] = round(time.time() - t0, 1)
            return st
    return {"error": f"client timeout after {timeout}s", "_wall_seconds": timeout}


def run_lane(lane: str, endpoint: str, gpu: str, image: str | None, extra: dict):
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        sys.exit("RUNPOD_API_KEY not set (source runpod-stack/.env)")
    os.makedirs(RESULTS, exist_ok=True)

    payload = {"input": {
        "prompt": PROMPT, "negative_prompt": NEGATIVE, "seed": SEED,
        "num_frames": NUM_FRAMES, "width": WIDTH, "height": HEIGHT, **extra,
    }}
    if image:
        if image.startswith("http"):
            payload["input"]["image_url"] = image
        else:
            with open(image, "rb") as f:
                payload["input"]["image_base64"] = base64.b64encode(f.read()).decode()

    print(f"== {lane} -> {endpoint} ({gpu}) ==", flush=True)
    res = call(endpoint, payload, api_key)
    wall = res.get("_wall_seconds", 0)
    out = (res.get("output") or {}) if isinstance(res.get("output"), dict) else {}

    rate = GPU_RATES.get(gpu, 0.0)
    record = {
        "lane": lane, "endpoint": endpoint, "gpu": gpu, "rate_usd_hr": rate,
        "status": res.get("status", "ERROR"),
        "wall_seconds": wall,
        "cost_usd": round(wall / 3600 * rate, 4),
        "generation_seconds": out.get("generation_seconds"),
        "model": out.get("model"), "dtype": out.get("dtype"),
        "frames": out.get("frames"), "fps": out.get("fps"),
        "width": out.get("width"), "height": out.get("height"),
        "steps": out.get("steps"),
        "error": res.get("error") or out.get("error"),
        "prompt": PROMPT, "seed": SEED,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    vid = out.get("video_url", "")
    if vid.startswith("data:"):
        mp4 = base64.b64decode(vid.split(",", 1)[1])
        path = os.path.join(RESULTS, f"{lane}.mp4")
        with open(path, "wb") as f:
            f.write(mp4)
        record["mp4"] = path
        record["mp4_bytes"] = len(mp4)
        print(f"  wrote {path} ({len(mp4)/1e6:.1f} MB)", flush=True)

    with open(os.path.join(RESULTS, f"{lane}.json"), "w") as f:
        json.dump(record, f, indent=2)
    print(f"  {record['status']} | {wall}s wall | ${record['cost_usd']} "
          f"| err={record['error']}", flush=True)
    return record


def replay():
    if not os.path.isdir(RESULTS):
        sys.exit("no results/ yet")
    rows = []
    for fn in sorted(os.listdir(RESULTS)):
        if fn.endswith(".json"):
            with open(os.path.join(RESULTS, fn)) as f:
                rows.append(json.load(f))
    if not rows:
        sys.exit("no result json files")
    print("| lane | model | gpu | dtype | steps | clip | wall | $/clip | status |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        secs = (f"{r['frames']}f@{r['fps']}fps"
                if r.get("frames") and r.get("fps") else "-")
        print(f"| {r['lane']} | {r.get('model') or '-'} | {r['gpu']} | "
              f"{r.get('dtype') or '-'} | {r.get('steps') or '-'} | {secs} | "
              f"{r['wall_seconds']}s | ${r['cost_usd']} | {r['status']} |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane")
    ap.add_argument("--endpoint")
    ap.add_argument("--gpu", default="A100")
    # Defaults to the exact still the 5B baseline was run against
    # (runpod-stack/sdxl_test.png, an SDXL-generated sunlit oak floor). Reusing it makes
    # the A14B comparison like-for-like on the input as well as the prompt -- a different
    # source image would quietly change the hardest part of an image-to-video test.
    ap.add_argument("--image", default=os.path.join(HERE, "..", "..",
                                                    "runpod-stack", "sdxl_test.png"),
                    help="path or url for image-to-video lanes")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--guidance", type=float)
    ap.add_argument("--replay", action="store_true")
    a = ap.parse_args()

    if a.replay:
        replay()
    else:
        if not (a.lane and a.endpoint):
            sys.exit("need --lane and --endpoint (or --replay)")
        extra = {}
        if a.steps:
            extra["num_inference_steps"] = a.steps
        if a.guidance is not None:
            extra["guidance_scale"] = a.guidance
        run_lane(a.lane, a.endpoint, a.gpu, a.image, extra)
