"""
Custom TTSCodec using NovaSR for audio upsampling.
Monkey-patches ncodec to use NovaSR instead of FlashSR.
"""

import gc
import torch
from huggingface_hub import snapshot_download
from mira.decoder_novasr import AudioDecoderNovaSR


class TTSCodecNovaSR:
    """
    TTSCodec that uses NovaSR for upsampling.
    Drop-in replacement for ncodec.codec.TTSCodec
    """

    def __init__(self):
        d_path = snapshot_download("YatharthS/MiraTTS")
        d_path = f"{d_path}/decoders"
        
        # Use NovaSR-based decoder
        self.audio_decoder = AudioDecoderNovaSR(d_path)
        
        # Use original encoder
        from ncodec.encoder.model import AudioEncoder
        self.audio_encoder = AudioEncoder(d_path)

    def encode(self, audio, encode_semantic=False, duration=8):
        if encode_semantic:
            speech_tokens, context_tokens = self.audio_encoder.encode(audio, True, duration=duration)
            return speech_tokens, context_tokens
        else:
            context_tokens = self.audio_encoder.encode(audio, False, duration=duration)
            return context_tokens

    def format_prompt(self, text, context_tokens, extra_tokens, semantic_tokens=None, transcript=None):
        if semantic_tokens:
            prompt = f"<|task_tts|><|start_text|>{text}<|end_text|><|context_audio_start|>{context_tokens}<|context_audio_end|><|prompt_speech_start|>{semantic_tokens}"
        else:
            prompt = f"<|task_tts|><|start_text|>{text}<|end_text|><|context_audio_start|>{context_tokens}<|context_audio_end|><|prompt_speech_start|>"
        return prompt

    def c_cache(self):
        gc.collect()
        torch.cuda.empty_cache()

    def decode(self, speech_tokens, context_tokens, test_var=None):
        wav = self.audio_decoder.detokenize(
            context_tokens,
            speech_tokens,
        )
        return wav

