# Hunyuan3D 2.1 -- QUARANTINED BY LICENCE. Benchmark comparator, not a production lane.
#
# READ THIS BEFORE DEPLOYING. Tencent's Hunyuan3D 2.1 Community Licence grants rights
# for a Territory defined as worldwide "excluding the territory of the European Union,
# United Kingdom and South Korea" (Tencent Hunyuan 3D 2.1 Community License, section 1(l)).
# The exclusion is not limited to running the model -- section 5(c) and the Acceptable
# Use Policy extend it to the *Output*. A mesh this endpoint generates may not be
# displayed to, or distributed in, those jurisdictions. For an agency that cannot
# enumerate where a client's customers are, that is a liability attached to every
# asset it produces.
#
# It also carries a 1,000,000 monthly-active-user trigger above which no rights are
# exercisable until Tencent grants a separate licence at its sole discretion. Note that
# figure: it is 1M, not the 100M that the Llama-style licences popularised, and it is
# two orders of magnitude easier to hit than most summaries imply.
#
# THIS IS A SEPARATE IMAGE ON PURPOSE. The same reasoning the sibling comfy worker used
# when it deleted the non-commercial VFI models from its own image rather than writing
# a comment: an operator picking a model from a dropdown cannot see a licence. Keeping
# Hunyuan3D out of the default 3D image means the production lane cannot reach it by
# accident. Deploy this only to run the benchmark, then park it at --workers-max 0.
#
# WHAT THIS ACTUALLY PRODUCES. ComfyUI core's Hunyuan3D 2.1 path is
# ImageOnlyCheckpointLoader -> KSampler -> VAEDecodeHunyuan3D -> VoxelToMesh -> SaveGLB.
# That is GEOMETRY ONLY -- there is no texture stage in core. Hunyuan3D 2.1's reputation
# as the texture/quality leader rests on its own paint pipeline, which is not in
# ComfyUI. Comparing this against TRELLIS.2 compares Hunyuan's shape to TRELLIS.2's
# shape-plus-PBR. Recorded in stack-docs/3D-BENCHMARK.md so the comparison is not
# misread as "TRELLIS.2 wins on texture".
FROM runpod/worker-comfyui:5.10.0-base

RUN comfy model download \
      --url https://huggingface.co/Comfy-Org/hunyuan3D_2.1_repackaged/resolve/main/hunyuan_3d_v2.1.safetensors \
      --relative-path models/checkpoints \
      --filename hunyuan_3d_v2.1.safetensors

# DELIBERATELY NOT IN .github/workflows/build.yml. A licence-restricted model should
# not be rebuilt and pushed on every commit to main -- CI publishing it to GHCR on a
# schedule is how it ends up deployed by someone who never read this header. Build it
# by hand when the benchmark needs refreshing, from the PARENT directory so it can
# share the handler and converter:
#
#   docker build -f runpod-own-worker/threed/hunyuan/Dockerfile \
#                -t ghcr.io/aspireco/runpod-threed-hunyuan-worker:latest \
#                runpod-own-worker/threed
#
# Hence the COPY paths below are relative to threed/, not to this directory.
COPY hunyuan/hunyuan3d_image_to_mesh.ui.json /workflows/hunyuan3d_image_to_mesh.ui.json
COPY build_api_workflows.py /build_api_workflows.py
COPY handler.py /handler.py

# Node 10 is the SaveGLB that terminates the graph.
RUN python /build_api_workflows.py \
      /workflows/hunyuan3d_image_to_mesh.ui.json \
      /workflows/hunyuan3d_image_to_mesh.api.json \
      10

RUN cd /comfyui && timeout 900 python main.py --quick-test-for-ci --cpu \
 && echo "comfyui boots clean"

RUN python -c "import ast; ast.parse(open('/handler.py').read()); print('handler parses OK')" \
 && python -c "import json; json.load(open('/workflows/hunyuan3d_image_to_mesh.api.json')); print('api workflow valid')"

ENV WORKFLOW_DIR=/workflows \
    PYTHONUNBUFFERED=1

CMD ["/start.sh"]
