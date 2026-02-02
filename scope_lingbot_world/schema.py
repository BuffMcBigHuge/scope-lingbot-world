"""LingBot-World pipeline configuration schema for Scope."""

from typing import ClassVar

from scope.core.pipelines.artifacts import HuggingfaceRepoArtifact
from scope.core.pipelines.base_schema import (
    BasePipelineConfig,
    ModeDefaults,
    height_field,
    width_field,
)


class LingBotWorldConfig(BasePipelineConfig):
    """Configuration for LingBot-World (image-to-video) using NF4 diffusion weights."""

    pipeline_id: ClassVar[str] = "lingbot_world"
    pipeline_name: ClassVar[str] = "LingBot-World (NF4)"
    pipeline_description: ClassVar[str] = (
        "Image-to-video world model (Wan2.1-based) with NF4 quantized diffusion weights"
    )
    pipeline_version: ClassVar[str] = "0.1.0"
    docs_url: ClassVar[str | None] = "https://github.com/Robbyant/lingbot-world"
    estimated_vram_gb: ClassVar[float | None] = 32.0

    # This pipeline requires model files.
    requires_models: ClassVar[bool] = True

    # Scope will download these into its model directory.
    # We need base assets (VAE + T5 + tokenizer) and NF4 diffusion weights.
    artifacts = [
        HuggingfaceRepoArtifact(
            repo_id="robbyant/lingbot-world-base-cam",
            files=[
                # VAE
                "Wan2.1_VAE.pth",
                # T5 encoder weights
                "models_t5_umt5-xxl-enc-bf16.pth",
                # Tokenizer directory
                "google",
            ],
        ),
        HuggingfaceRepoArtifact(
            repo_id="cahlen/lingbot-world-base-cam-nf4",
            files=[
                "high_noise_model_bnb_nf4/config.json",
                "high_noise_model_bnb_nf4/model.pt",
                "low_noise_model_bnb_nf4/config.json",
                "low_noise_model_bnb_nf4/model.pt",
                "README.md",
            ],
        ),
    ]

    supports_lora: ClassVar[bool] = False
    supports_vace: ClassVar[bool] = False
    supports_quantization: ClassVar[bool] = True
    recommended_quantization_vram_threshold: ClassVar[float | None] = 48.0

    # Scope modes: declare as video-mode pipeline expecting 1 input frame (the starting image).
    modes: ClassVar[dict[str, ModeDefaults]] = {
        "video": ModeDefaults(default=True, input_size=1)
    }

    supports_prompts: ClassVar[bool] = True

    # Output resolution defaults. LingBot-World examples are 480x832 or 720x1280.
    height: int = height_field(default=480)
    width: int = width_field(default=832)

    # LingBot-World parameters
    frame_num: int = 81  # must be 4n+1 per upstream
    sampling_steps: int = 40
    guide_scale: float = 5.0
    shift: float = 5.0

    # Whether to keep T5 on CPU to reduce VRAM
    t5_cpu: bool = True

    # If true, ignore base_seed and randomize seed per call.
    randomize_seed: bool = False
