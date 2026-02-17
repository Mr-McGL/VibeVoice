"""Compatibility loader for long-form VibeVoice TTS inference models.

This repository does not ship the full non-streaming TTS inference graph directly,
but many users still rely on the historical import path:

    vibevoice.modular.modeling_vibevoice_inference.VibeVoiceForConditionalGenerationInference

To keep those scripts working, this module provides a lightweight adapter that
loads whichever backend the checkpoint supports:

* Native streaming backend from this repository (``model_type=vibevoice_streaming``)
* Remote-code backend from the checkpoint repository (``trust_remote_code=True``)
"""

from __future__ import annotations

from typing import Any

from transformers import AutoConfig, AutoModelForCausalLM

from .modeling_vibevoice_streaming_inference import (
    VibeVoiceStreamingForConditionalGenerationInference,
)


class VibeVoiceForConditionalGenerationInference:
    """Backward-compatible factory for VibeVoice TTS inference models.

    Notes:
        This class intentionally behaves as a factory. ``from_pretrained`` returns
        an initialized model instance from either local streaming code or the
        checkpoint's remote-code implementation.
    """

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args: Any, **kwargs: Any):
        """Load a VibeVoice TTS inference model.

        Args:
            pretrained_model_name_or_path: Local path or model hub id.
            *model_args: Forwarded to Transformers ``from_pretrained``.
            **kwargs: Forwarded to Transformers ``from_pretrained``.

        Returns:
            A model object exposing the inference API (e.g., ``generate``,
            ``set_ddpm_inference_steps``).
        """

        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Streaming checkpoints can be handled by the local implementation.
        if getattr(config, "model_type", None) == "vibevoice_streaming":
            return VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                **kwargs,
            )

        # Long-form checkpoints are loaded from their own modeling code.
        kwargs.setdefault("trust_remote_code", True)
        return AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            **kwargs,
        )
