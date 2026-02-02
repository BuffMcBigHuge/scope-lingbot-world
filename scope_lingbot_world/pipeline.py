"""LingBot-World (NF4) pipeline implementation for Scope.

This is a best-effort wrapper around LingBot-World's image-to-video generation.

Key points:
- Scope passes input frames as uint8 THWC tensors inside kwargs["video"].
- LingBot-World expects a PIL image + prompt (+ optional camera pose folder).
- We load the NF4 diffusion weights from the cahlen HF repo (model.pt state dicts)
  and base assets (VAE + T5 + tokenizer) from the robbyant HF repo.

We intentionally keep the implementation self-contained and avoid torchrun/FSDP.
"""

from __future__ import annotations

import gc
import logging
import os
import random
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from scope.core.config import get_model_file_path
from scope.core.pipelines.interface import Pipeline

from .schema import LingBotWorldConfig

if TYPE_CHECKING:
    from scope.core.pipelines.schema import BasePipelineConfig

logger = logging.getLogger(__name__)


def _replace_linear_with_nf4(model: torch.nn.Module) -> tuple[torch.nn.Module, int]:
    """Replace all nn.Linear layers with bitsandbytes Linear4bit (NF4).

    This mirrors the approach in cahlen/lingbot-world-base-cam-nf4/quantize_bnb.py.

    Note: we don't try to be clever about skipping layers; we follow the upstream
    quantization approach (Linear only).
    """

    import bitsandbytes as bnb

    replaced = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, torch.nn.Linear):
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = model.get_submodule(parent_name) if parent_name else model

            nf4_linear = bnb.nn.Linear4bit(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                compute_dtype=torch.bfloat16,
                compress_statistics=True,
                quant_type="nf4",
            )

            # Placeholder params (will be overwritten by load_state_dict)
            nf4_linear.weight = bnb.nn.Params4bit(
                torch.empty_like(module.weight.data),
                requires_grad=False,
                compress_statistics=True,
                quant_type="nf4",
            )
            if module.bias is not None:
                nf4_linear.bias = torch.nn.Parameter(torch.empty_like(module.bias.data))

            setattr(parent, child_name, nf4_linear)
            replaced += 1

    return model, replaced


def _load_nf4_wan_model(model_dir: Path) -> torch.nn.Module:
    """Load a WanModel from a quantized folder containing config.json + model.pt."""

    import bitsandbytes  # noqa: F401 (import verifies availability)

    from wan.modules.model import WanModel

    # Load config and instantiate model
    cfg_dict = WanModel.load_config(str(model_dir))
    model = WanModel.from_config(cfg_dict)

    # Ensure module types match the quantized checkpoint
    model, replaced = _replace_linear_with_nf4(model)
    logger.info(f"LingBot-World: replaced {replaced} Linear layers with NF4 Linear4bit")

    # Load state dict
    state_path = model_dir / "model.pt"
    state = torch.load(state_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning(f"LingBot-World: missing keys when loading {state_path}: {missing[:10]}...")
    if unexpected:
        logger.warning(
            f"LingBot-World: unexpected keys when loading {state_path}: {unexpected[:10]}..."
        )

    model.eval().requires_grad_(False)
    return model


class _WanI2V_NF4:
    """Single-GPU LingBot-World image-to-video with pre-quantized NF4 diffusion models."""

    def __init__(
        self,
        base_dir: Path,
        nf4_dir: Path,
        device: torch.device,
        t5_cpu: bool = True,
    ):
        from wan.configs.wan_i2v_A14B import i2v_A14B as cfg
        from wan.modules.t5 import T5EncoderModel
        from wan.modules.vae2_1 import Wan2_1_VAE

        self.device = device
        self.cfg = cfg
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = cfg.num_train_timesteps
        self.boundary = cfg.boundary
        self.param_dtype = cfg.param_dtype
        self.vae_stride = cfg.vae_stride
        self.patch_size = cfg.patch_size
        self.sample_neg_prompt = cfg.sample_neg_prompt

        # T5 encoder (often kept on CPU to save VRAM)
        self.text_encoder = T5EncoderModel(
            text_len=cfg.text_len,
            dtype=cfg.t5_dtype,
            device=torch.device("cpu"),
            checkpoint_path=str(base_dir / cfg.t5_checkpoint),
            tokenizer_path=str(base_dir / cfg.t5_tokenizer),
            shard_fn=None,
        )

        # VAE (GPU)
        self.vae = Wan2_1_VAE(
            vae_pth=str(base_dir / cfg.vae_checkpoint),
            device=self.device,
        )

        # NF4 diffusion models
        # The HF repo stores them as:
        #  - low_noise_model_bnb_nf4/
        #  - high_noise_model_bnb_nf4/
        self.low_noise_model = _load_nf4_wan_model(nf4_dir / "low_noise_model_bnb_nf4")
        self.high_noise_model = _load_nf4_wan_model(nf4_dir / "high_noise_model_bnb_nf4")

        # Keep both on CPU until used (reduces idle VRAM)
        self.low_noise_model.to("cpu")
        self.high_noise_model.to("cpu")

    def _prepare_model_for_timestep(self, t: torch.Tensor, boundary: float):
        if t.item() >= boundary:
            required_name = "high_noise_model"
            offload_name = "low_noise_model"
        else:
            required_name = "low_noise_model"
            offload_name = "high_noise_model"

        required = getattr(self, required_name)
        offload = getattr(self, offload_name)

        # Offload unused model
        try:
            if next(offload.parameters()).device.type == "cuda":
                offload.to("cpu")
                torch.cuda.empty_cache()
        except StopIteration:
            pass

        # Load required model
        try:
            if next(required.parameters()).device.type == "cpu":
                required.to(self.device)
        except StopIteration:
            pass

        return required

    @torch.inference_mode()
    def generate(
        self,
        *,
        prompt: str,
        init_image_pil,
        max_area: int,
        frame_num: int,
        sampling_steps: int,
        guide_scale: float,
        shift: float,
        seed: int,
        action_path: str | None = None,
        negative_prompt: str | None = None,
    ) -> torch.Tensor:
        """Return frames as tensor shaped [3, F, H, W] in [-1,1] (Wan VAE space)."""

        import numpy as np
        import torchvision.transforms.functional as TF
        from einops import rearrange
        from wan.utils.cam_utils import (
            compute_relative_poses,
            get_Ks_transformed,
            get_plucker_embeddings,
            interpolate_camera_poses,
        )
        from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

        if negative_prompt is None or negative_prompt == "":
            negative_prompt = self.sample_neg_prompt

        # Frame count constraint from upstream (4n + 1)
        if (frame_num - 1) % 4 != 0:
            raise ValueError("frame_num must satisfy frame_num = 4n + 1")

        if action_path is not None:
            c2ws = np.load(os.path.join(action_path, "poses.npy"))
            len_c2ws = ((len(c2ws) - 1) // 4) * 4 + 1
            frame_num = min(frame_num, len_c2ws)
            c2ws = c2ws[:frame_num]

        # Prepare init image tensor in [-1, 1]
        img_tensor = TF.to_tensor(init_image_pil).sub_(0.5).div_(0.5).to(self.device)

        F = frame_num
        h0, w0 = img_tensor.shape[1:]
        aspect_ratio = h0 / w0

        lat_h = round(
            (max_area * aspect_ratio) ** 0.5
            // self.vae_stride[1]
            // self.patch_size[1]
            * self.patch_size[1]
        )
        lat_w = round(
            (max_area / aspect_ratio) ** 0.5
            // self.vae_stride[2]
            // self.patch_size[2]
            * self.patch_size[2]
        )

        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]
        lat_f = (F - 1) // self.vae_stride[0] + 1
        max_seq_len = lat_f * lat_h * lat_w // (self.patch_size[1] * self.patch_size[2])

        seed_g = torch.Generator(device=self.device).manual_seed(seed)
        noise = torch.randn(
            16,
            (F - 1) // self.vae_stride[0] + 1,
            lat_h,
            lat_w,
            dtype=torch.float32,
            generator=seed_g,
            device=self.device,
        )

        msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat(
            [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]],
            dim=1,
        )
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        # Encode text (T5)
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([prompt], self.device)
            context_null = self.text_encoder([negative_prompt], self.device)
            self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([prompt], torch.device("cpu"))
            context_null = self.text_encoder([negative_prompt], torch.device("cpu"))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        # Camera conditioning
        dit_cond_dict = None
        if action_path is not None:
            Ks = torch.from_numpy(np.load(os.path.join(action_path, "intrinsics.npy"))).float()
            Ks = get_Ks_transformed(Ks, 480, 832, h, w, h, w)
            Ks = Ks[0]

            len_c2ws = len(c2ws)
            c2ws_infer = interpolate_camera_poses(
                src_indices=np.linspace(0, len_c2ws - 1, len_c2ws),
                src_rot_mat=c2ws[:, :3, :3],
                src_trans_vec=c2ws[:, :3, 3],
                tgt_indices=np.linspace(0, len_c2ws - 1, int((len_c2ws - 1) // 4) + 1),
            )
            c2ws_infer = compute_relative_poses(c2ws_infer, framewise=True)
            Ks = Ks.repeat(len(c2ws_infer), 1)

            c2ws_infer = c2ws_infer.to(self.device)
            Ks = Ks.to(self.device)
            c2ws_plucker_emb = get_plucker_embeddings(c2ws_infer, Ks, h, w)
            c2ws_plucker_emb = rearrange(
                c2ws_plucker_emb,
                "f (h c1) (w c2) c -> (f h w) (c c1 c2)",
                c1=int(h // lat_h),
                c2=int(w // lat_w),
            )
            c2ws_plucker_emb = c2ws_plucker_emb[None, ...]
            c2ws_plucker_emb = rearrange(
                c2ws_plucker_emb,
                "b (f h w) c -> b c f h w",
                f=lat_f,
                h=lat_h,
                w=lat_w,
            ).to(self.param_dtype)
            dit_cond_dict = {"c2ws_plucker_emb": c2ws_plucker_emb.chunk(1, dim=0)}

        # Encode image and mask
        y = self.vae.encode(
            [
                torch.concat(
                    [
                        torch.nn.functional.interpolate(
                            img_tensor[None].cpu(), size=(h, w), mode="bicubic"
                        ).transpose(0, 1),
                        torch.zeros(3, F - 1, h, w),
                    ],
                    dim=1,
                ).to(self.device)
            ]
        )[0]
        y = torch.concat([msk, y])

        # Diffusion sampling
        boundary = self.boundary * self.num_train_timesteps
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False,
        )
        scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
        timesteps = scheduler.timesteps

        latent = noise

        arg_c = {
            "context": [context[0]],
            "seq_len": max_seq_len,
            "y": [y],
            "dit_cond_dict": dit_cond_dict,
        }
        arg_null = {
            "context": context_null,
            "seq_len": max_seq_len,
            "y": [y],
            "dit_cond_dict": dit_cond_dict,
        }

        # pre-load first model
        first_name = "high_noise_model" if timesteps[0].item() >= boundary else "low_noise_model"
        getattr(self, first_name).to(self.device)

        with torch.amp.autocast("cuda", dtype=self.param_dtype), torch.no_grad():
            for t in timesteps:
                latent_model_input = [latent.to(self.device)]
                timestep = torch.stack([t]).to(self.device)

                model = self._prepare_model_for_timestep(t, boundary)
                noise_pred_cond = model(latent_model_input, t=timestep, **arg_c)[0]
                noise_pred_uncond = model(latent_model_input, t=timestep, **arg_null)[0]
                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond
                )

                latent = scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g,
                )[0].squeeze(0)

        # Offload models back to CPU
        self.low_noise_model.to("cpu")
        self.high_noise_model.to("cpu")
        torch.cuda.empty_cache()

        # Decode
        videos = self.vae.decode([latent])

        del noise
        del latent
        gc.collect()

        return videos[0]


class LingBotWorldPipeline(Pipeline):
    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return LingBotWorldConfig

    def __init__(
        self,
        height: int = 480,
        width: int = 832,
        frame_num: int = 81,
        sampling_steps: int = 40,
        guide_scale: float = 5.0,
        shift: float = 5.0,
        t5_cpu: bool = True,
        randomize_seed: bool = False,
        device: torch.device | None = None,
        base_dir: str | None = None,
        nf4_dir: str | None = None,
        **kwargs,
    ):
        # Accept/ignore unknown scope params for forward compatibility
        if kwargs:
            logger.debug(
                f"LingBotWorldPipeline ignoring unknown kwargs: {list(kwargs.keys())}"
            )

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.height = height
        self.width = width
        self.frame_num = frame_num
        self.sampling_steps = sampling_steps
        self.guide_scale = guide_scale
        self.shift = shift
        self.t5_cpu = t5_cpu
        self.randomize_seed = randomize_seed

        # Resolve model directories from Scope's model directory
        self.base_dir = Path(base_dir) if base_dir else get_model_file_path("lingbot-world-base-cam")
        self.nf4_dir = Path(nf4_dir) if nf4_dir else get_model_file_path("lingbot-world-base-cam-nf4")

        self._impl: _WanI2V_NF4 | None = None

    def _get_impl(self) -> _WanI2V_NF4:
        if self._impl is None:
            self._impl = _WanI2V_NF4(
                base_dir=self.base_dir,
                nf4_dir=self.nf4_dir,
                device=self.device,
                t5_cpu=self.t5_cpu,
            )
        return self._impl

    def __call__(self, **kwargs) -> dict:
        """Scope entrypoint."""

        # Prompt handling: Scope uses a list of prompt dicts under `prompts`.
        prompts = kwargs.get("prompts")
        if prompts and isinstance(prompts, list) and len(prompts) > 0:
            prompt = prompts[0].get("text") or ""
        else:
            prompt = kwargs.get("prompt") or ""
        if not prompt:
            prompt = "a cinematic first-person exploration through a rich environment"

        # Seed
        base_seed = int(kwargs.get("seed", kwargs.get("base_seed", 42)))
        if self.randomize_seed:
            seed = random.randint(0, 2**31 - 1)
        else:
            seed = base_seed

        # Input image: Scope passes `video` as list of frames (uint8 THWC)
        input_video = kwargs.get("video")
        if not input_video or not isinstance(input_video, list) or len(input_video) < 1:
            raise ValueError(
                "LingBotWorldPipeline expects a single input frame via kwargs['video']"
            )

        first_frame = input_video[0]
        if not isinstance(first_frame, torch.Tensor):
            raise ValueError("kwargs['video'][0] must be a torch.Tensor")

        # first_frame shape: (1, H, W, C), uint8 in [0,255]
        frame = first_frame
        if frame.ndim == 4 and frame.shape[0] == 1:
            frame = frame[0]

        # Convert to PIL image (RGB)
        from PIL import Image

        frame_u8 = frame.detach().to(torch.uint8).cpu().numpy()
        pil_image = Image.fromarray(frame_u8, mode="RGB")

        max_area = int(self.height * self.width)

        impl = self._get_impl()
        video_chw = impl.generate(
            prompt=prompt,
            init_image_pil=pil_image,
            max_area=max_area,
            frame_num=int(self.frame_num),
            sampling_steps=int(self.sampling_steps),
            guide_scale=float(self.guide_scale),
            shift=float(self.shift),
            seed=seed,
            action_path=None,
        )

        # Convert [3, F, H, W] in [-1,1] -> [F, H, W, 3] in [0,1]
        video_thwc = ((video_chw + 1.0) / 2.0).clamp(0.0, 1.0).permute(1, 2, 3, 0)
        return {"video": video_thwc}
