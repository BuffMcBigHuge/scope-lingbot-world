"""LingBot-World (NF4) image-to-video pipeline plugin for Daydream Scope."""

import scope.core

from .pipeline import LingBotWorldPipeline
from .schema import LingBotWorldConfig


@scope.core.hookimpl
def register_pipelines(register):
    register(LingBotWorldPipeline)


__all__ = ["LingBotWorldPipeline", "LingBotWorldConfig"]
