from typing import Dict, List, Optional

import torch
from .modeling_vibevoice import VibeVoiceForConditionalGeneration, VibeVoiceGenerationOutput


class VibeVoiceForConditionalGenerationInference(VibeVoiceForConditionalGeneration):
    """Inference wrapper for offline VibeVoice-TTS with diffusion speech decoding."""

    def __init__(self, config):
        super().__init__(config)
        self.set_ddpm_inference_steps()

    def set_ddpm_inference_steps(self, num_steps: Optional[int] = None):
        self.ddpm_inference_steps = num_steps or self.config.diffusion_head_config.ddpm_num_inference_steps

    @torch.no_grad()
    def sample_speech_tokens(self, condition, neg_condition, cfg_scale: float = 1.0):
        self.model.noise_scheduler.set_timesteps(self.ddpm_inference_steps)
        condition = torch.cat([condition, neg_condition], dim=0).to(self.model.prediction_head.device)
        speech = torch.randn(condition.shape[0], self.config.acoustic_vae_dim, device=condition.device, dtype=condition.dtype)
        for t in self.model.noise_scheduler.timesteps:
            half = speech[: len(speech) // 2]
            combined = torch.cat([half, half], dim=0)
            eps = self.model.prediction_head(combined, t.repeat(combined.shape[0]).to(combined), condition=condition)
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            speech = self.model.noise_scheduler.step(eps, t, speech).prev_sample
        return speech[: len(speech) // 2]

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        speech_tensors: Optional[torch.FloatTensor] = None,
        speech_masks: Optional[torch.BoolTensor] = None,
        speech_input_mask: Optional[torch.BoolTensor] = None,
        tokenizer=None,
        max_new_tokens: Optional[int] = None,
        cfg_scale: float = 1.0,
        return_speech: bool = True,
        generation_config: Optional[Dict] = None,
        verbose: bool = False,
        refresh_negative: bool = True,
        **kwargs,
    ):
        if tokenizer is None:
            raise ValueError("tokenizer is required for VibeVoice TTS generation")

        if input_ids.shape[0] != 1:
            raise ValueError("Only batch size 1 is currently supported")

        device = input_ids.device
        do_sample = False
        temperature = 1.0
        if isinstance(generation_config, dict):
            do_sample = bool(generation_config.get("do_sample", False))
            temperature = float(generation_config.get("temperature", 1.0))

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device)

        if max_new_tokens is None:
            max_new_tokens = self.config.decoder_config.max_position_embeddings - input_ids.shape[1]

        speech_pad_id = tokenizer.speech_diffusion_id
        speech_end_id = tokenizer.speech_end_id
        neg_text_input_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")

        # Prefill condition branch with voice prompt.
        prefill_out = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            speech_tensors=speech_tensors,
            speech_masks=speech_masks,
            acoustic_input_mask=speech_input_mask,
            use_cache=True,
            return_dict=True,
            speech_type="audio",
        )
        cur_ids = input_ids
        cur_attention = attention_mask

        # Unconditional branch for CFG.
        neg_ids = torch.tensor([[neg_text_input_id]], dtype=torch.long, device=device)
        neg_attention = torch.ones_like(neg_ids, device=device)
        neg_out = self.forward(
            input_ids=neg_ids,
            attention_mask=neg_attention,
            use_cache=True,
            return_dict=True,
            speech_type="audio",
        )
        neg_past = neg_out.past_key_values
        neg_step = neg_out

        generated_ids: List[int] = []
        audio_chunks: List[torch.Tensor] = []
        out = prefill_out

        for _ in range(max_new_tokens):
            logits = out.logits[:, -1, :]
            if do_sample:
                probs = torch.softmax(logits / max(temperature, 1e-5), dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            token_id = int(next_token.item())
            generated_ids.append(token_id)
            cur_ids = torch.cat([cur_ids, next_token], dim=-1)
            cur_attention = torch.cat([cur_attention, torch.ones((1, 1), dtype=cur_attention.dtype, device=device)], dim=-1)

            if token_id == speech_end_id:
                break

            if token_id != speech_pad_id:
                # Keep negative branch in sync with normal text tokens.
                if refresh_negative:
                    neg_in = next_token
                    neg_attention = torch.cat([neg_attention, torch.ones((1, 1), dtype=neg_attention.dtype, device=device)], dim=-1)
                    neg_step = self.forward(
                        input_ids=neg_in,
                        attention_mask=neg_attention,
                        past_key_values=neg_past,
                        use_cache=True,
                        return_dict=True,
                        speech_type="audio",
                    )
                    neg_past = neg_step.past_key_values
                out = self.forward(
                    input_ids=next_token,
                    attention_mask=cur_attention,
                    past_key_values=out.past_key_values,
                    use_cache=True,
                    return_dict=True,
                    speech_type="audio",
                )
                continue

            if refresh_negative:
                neg_condition = neg_step.last_hidden_state[:, -1, :]
            else:
                neg_condition = neg_out.last_hidden_state[:, -1, :]

            pos_condition = out.last_hidden_state[:, -1, :]
            speech_latent = self.sample_speech_tokens(pos_condition, neg_condition, cfg_scale=cfg_scale).unsqueeze(1)
            scaled_latent = speech_latent / self.model.speech_scaling_factor.to(speech_latent.device) - self.model.speech_bias_factor.to(speech_latent.device)
            if return_speech:
                audio_chunk = self.model.acoustic_tokenizer.decode(
                    scaled_latent.to(self.model.acoustic_tokenizer.device),
                    use_cache=False,
                    debug=False,
                )[0]
                audio_chunks.append(audio_chunk)

            # Replace last speech placeholder with latent embedding for future context.
            one_token_attention = torch.ones((1, 1), dtype=cur_attention.dtype, device=device)
            out_with_speech = self.forward(
                input_ids=next_token,
                attention_mask=cur_attention,
                past_key_values=out.past_key_values,
                speech_tensors=speech_latent,
                speech_masks=torch.ones((1, 1), dtype=torch.bool, device=device),
                acoustic_input_mask=torch.ones((1, 1), dtype=torch.bool, device=device),
                use_cache=True,
                return_dict=True,
                speech_type="vae",
            )
            out = out_with_speech

            if refresh_negative:
                neg_step = self.forward(
                    input_ids=next_token,
                    attention_mask=torch.cat([neg_attention, one_token_attention], dim=-1),
                    past_key_values=neg_past,
                    speech_tensors=speech_latent,
                    speech_masks=torch.ones((1, 1), dtype=torch.bool, device=device),
                    acoustic_input_mask=torch.ones((1, 1), dtype=torch.bool, device=device),
                    use_cache=True,
                    return_dict=True,
                    speech_type="vae",
                )
                neg_attention = torch.cat([neg_attention, one_token_attention], dim=-1)
                neg_past = neg_step.past_key_values

        sequences = torch.cat([input_ids, torch.tensor([generated_ids], dtype=torch.long, device=device)], dim=-1)
        speech_outputs = None
        if return_speech:
            speech_outputs = [torch.cat(audio_chunks, dim=-1) if audio_chunks else torch.zeros(0, device=device)]

        return VibeVoiceGenerationOutput(sequences=sequences, speech_outputs=speech_outputs)


__all__ = ["VibeVoiceForConditionalGenerationInference"]
