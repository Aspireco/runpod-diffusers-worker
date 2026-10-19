"""Run at image build time: fail the BUILD, not a billed GPU (a file, because ACR's dependency
scanner misreads a Dockerfile RUN line that begins -- after its `\` continuation -- with `from`;
every other worker in this repo carries this same note).

Unlike audio-clean's build_check.py (weights baked into the image, so it does a real load),
this lane's weights live on the Azure Files share (see Dockerfile/README.md) and are not present
at ACR build time -- there is no GPU on the build agent either. So this proves what CAN be proven
without them: the `--no-deps` install pulled a real, importable `acestep` package (not a stub or a
failed git checkout), none of the ~59 mixin modules `acestep.handler` composes reach for something
this image did not install (gradio/fastapi/lightning/tensorboard/peft/lycoris-lora/mlx/nano-vllm --
all absent on purpose, see the Dockerfile's own comment), `AceStepHandler()` constructs on CPU with
no checkpoint present, and the exact `GenerationParams`/`GenerationConfig` fields this handler.py
passes still exist on the installed version (a silent rename here would otherwise surface as a
`TypeError` on the very first real job instead of here).
"""
import torch

assert "cu13" not in torch.__version__, f"torch replaced: {torch.__version__}"

from acestep.handler import AceStepHandler  # noqa: E402
from acestep.inference import GenerationConfig, GenerationParams, generate_music  # noqa: E402,F401

# A real construct, not just an import: AceStepHandler.__init__ only sets attributes (model=None,
# vae=None, ...) and touches a local .cache/ path for its progress-estimates file -- no checkpoint,
# no CUDA -- so this runs clean on the ACR build agent and still proves the 59-mixin MRO resolves.
_handler = AceStepHandler()
assert _handler.model is None and _handler.vae is None, "unexpected pre-init state"

# The exact call shape handler.py uses, with real values, so a dataclass field this project renamed
# or removed (e.g. `thinking`, `shift`, `use_cot_metas`) fails the BUILD instead of the first job.
_params = GenerationParams(
    task_type="text2music",
    caption="a short instrumental test",
    lyrics="[Instrumental]",
    instrumental=True,
    duration=10.0,
    inference_steps=8,
    guidance_scale=1.0,
    shift=3.0,
    seed=-1,
    thinking=False,
    use_cot_metas=False,
    use_cot_caption=False,
    use_cot_language=False,
    reference_audio=None,
)
_config = GenerationConfig(batch_size=1, use_random_seed=True, audio_format="wav")
assert _params.shift == 3.0 and _config.audio_format == "wav"

print("acestep 1.5 import + AceStepHandler() construct + GenerationParams/GenerationConfig OK")
