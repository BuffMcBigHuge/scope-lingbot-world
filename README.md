# scope-lingbot-world

A **Daydream Scope** plugin that adds **LingBot-World** image-to-video generation, using the **NF4 (4-bit) quantized** diffusion models from:

- <https://huggingface.co/cahlen/lingbot-world-base-cam-nf4>

This plugin is designed to work with Scope's plugin system (Pluggy entrypoints).

## What this plugin provides

- Pipeline id: `lingbot_world`
- Mode: `video` (expects a single input frame; uses it as the starting image)
- Output: a generated video tensor (`THWC`, float in `[0,1]`) under the `video` key.

## Model artifacts

LingBot-World's **quantized diffusion models** are distributed in a separate repo from the **base assets** (VAE + T5 + tokenizer files). You will need both:

1) **Base model assets** (VAE + T5 + tokenizer):
- <https://huggingface.co/robbyant/lingbot-world-base-cam>

2) **NF4 diffusion weights**:
- <https://huggingface.co/cahlen/lingbot-world-base-cam-nf4>

Scope should download these automatically via the pipeline's `artifacts` list.

## Install

```bash
pip install scope-lingbot-world
```

(Or install from GitHub while iterating.)

## Notes / assumptions

- The quantized weights in `cahlen/lingbot-world-base-cam-nf4` are stored as `model.pt` state dicts. This plugin reconstructs the model architecture from the bundled `config.json`, replaces `nn.Linear` layers with `bitsandbytes.nn.Linear4bit`, then loads the state dict.
- This plugin intentionally does **not** attempt multi-GPU / torchrun / FSDP. It is meant as a single-GPU pipeline wrapper for Scope.
- LingBot-World's upstream codebase is Wan2.1-based and not authored by Daydream.

## License

MIT
