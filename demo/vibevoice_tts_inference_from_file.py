"""Long-form TTS inference demo with advanced generation options.

Example:
python demo/vibevoice_tts_inference_from_file.py \
  --model models/microsoft/VibeVoice-1.5B \
  --voice-dir demo/voices/wav \
  --voice SpeakerA=alice.wav \
  --text-file demo/text_examples/1p_vibevoice.txt
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoTokenizer

from vibevoice.modular.modeling_vibevoice_inference import (
    VibeVoiceForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor


def parse_script(text: str, voices: dict[str, str], voice_dir: Path) -> tuple[str, dict[str, str]]:
    speaker_wavs = {name: str((voice_dir / wav).resolve()) for name, wav in voices.items()}
    script = text
    speaker_mapping: dict[str, str] = {}

    for index, (speaker_name, wav_path) in enumerate(speaker_wavs.items(), start=1):
        pattern = rf"(?m)^\s*\[{re.escape(speaker_name)}\]\s*"
        if re.search(pattern, script):
            script = re.sub(pattern, f"Speaker {index}: ", script)
        speaker_mapping[f"Speaker {index}"] = wav_path

    return script.strip() + "\n", speaker_mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--text-file", type=Path, required=True)
    parser.add_argument("--voice-dir", type=Path, required=True)
    parser.add_argument(
        "--voice",
        action="append",
        default=[],
        help="Mapping NAME=voice.wav (repeatable).",
    )
    parser.add_argument("--output", type=Path, default=Path("output.wav"))
    parser.add_argument("--ddpm-steps", type=int, default=10)
    parser.add_argument("--cfg-scale", type=float, default=1.9)
    args = parser.parse_args()

    voices: dict[str, str] = {}
    for item in args.voice:
        if "=" not in item:
            raise ValueError(f"Invalid --voice '{item}'. Use NAME=file.wav")
        key, value = item.split("=", 1)
        voices[key] = value

    text = args.text_file.read_text(encoding="utf-8")
    script, speaker_wavs = parse_script(text, voices, args.voice_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    processor = VibeVoiceProcessor.from_pretrained(args.model)
    tokenizer = processor.tokenizer or AutoTokenizer.from_pretrained(args.model)

    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        args.model,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    ).to(device)
    model.eval()
    model.set_ddpm_inference_steps(num_steps=args.ddpm_steps)

    inputs = processor(
        text=script,
        voice_samples=[speaker_wavs[k] for k in sorted(speaker_wavs.keys())],
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            tokenizer=tokenizer,
            cfg_scale=args.cfg_scale,
            return_speech=True,
            generation_config={"do_sample": False},
            refresh_negative=True,
            verbose=False,
        )

    audio = out.speech_outputs[0].detach().cpu().float().numpy().squeeze()
    sf.write(args.output, audio, 24000)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
