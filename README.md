# scope-lingbot-world

A **Daydream Scope** plugin that adds **LingBot-World** image-to-video generation, using the **NF4 (4-bit) quantized** diffusion models from:

- <https://huggingface.co/cahlen/lingbot-world-base-cam-nf4>

This plugin is designed to work with Scope's plugin system (Pluggy entrypoints).

## What this plugin provides

- Pipeline id: `lingbot_world`
- Mode: `video` (expects a single input frame; uses it as the starting image)
- Output: a generated video tensor (`THWC`, float in `[0,1]`) under the `video` key.

## Model artifacts

LingBot-World’s **NF4 diffusion weights** are distributed separately from the **base assets** (VAE + T5 + tokenizer). You need both:

1) Base assets:
- <https://huggingface.co/robbyant/lingbot-world-base-cam>

2) NF4 diffusion weights:
- <https://huggingface.co/cahlen/lingbot-world-base-cam-nf4>

Scope should download these automatically via the pipeline’s `artifacts` list.

### HuggingFace token

If you hit 401/403 or rate limits, set a read token:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

## Install

Follow Scope’s **manual installation** flow (plugin support is preview/CLI-only right now).

Install the plugin within the `scope` directory:

```bash
DAYDREAM_SCOPE_PREVIEW=1 uv run daydream-scope install git+https://github.com/BuffMcBigHuge/scope-lingbot-world.git
```

Confirm that the plugin is installed:

```bash
DAYDREAM_SCOPE_PREVIEW=1 uv run daydream-scope plugins
```

Confirm that the `lingbot_world` pipeline is available:

```bash
DAYDREAM_SCOPE_PREVIEW=1 uv run daydream-scope pipelines
```

## Upgrade

```bash
DAYDREAM_SCOPE_PREVIEW=1 uv run daydream-scope install --upgrade git+https://github.com/BuffMcBigHuge/scope-lingbot-world.git
```

## Notes / assumptions

- The quantized weights in `cahlen/lingbot-world-base-cam-nf4` are stored as `model.pt` state dicts. This plugin reconstructs the model architecture from the bundled `config.json`, replaces `nn.Linear` layers with `bitsandbytes.nn.Linear4bit`, then loads the state dict.
- This plugin intentionally does **not** attempt multi-GPU / torchrun / FSDP. It is meant as a single-GPU pipeline wrapper for Scope.
- LingBot-World is Wan2.1-based; upstream repo: <https://github.com/Robbyant/lingbot-world>

## License

MIT
