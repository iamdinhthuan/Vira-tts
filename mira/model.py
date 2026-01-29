import gc
import torch
from itertools import cycle
from ncodec.codec import TTSCodec
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig

from mira.utils import clear_cache, split_text


def crossfade_audio(audio1: torch.Tensor, audio2: torch.Tensor, crossfade_samples: int) -> torch.Tensor:
    """
    Crossfade between two audio tensors.

    Args:
        audio1: First audio tensor (1D)
        audio2: Second audio tensor (1D)
        crossfade_samples: Number of samples for crossfade

    Returns:
        Concatenated audio with crossfade applied
    """
    if crossfade_samples <= 0 or len(audio1) < crossfade_samples or len(audio2) < crossfade_samples:
        return torch.cat([audio1, audio2], dim=0)

    # Create fade curves
    fade_out = torch.linspace(1.0, 0.0, crossfade_samples, device=audio1.device, dtype=audio1.dtype)
    fade_in = torch.linspace(0.0, 1.0, crossfade_samples, device=audio2.device, dtype=audio2.dtype)

    # Apply crossfade
    audio1_end = audio1[-crossfade_samples:] * fade_out
    audio2_start = audio2[:crossfade_samples] * fade_in
    crossfaded = audio1_end + audio2_start

    # Concatenate: audio1[:-crossfade] + crossfaded + audio2[crossfade:]
    result = torch.cat([
        audio1[:-crossfade_samples],
        crossfaded,
        audio2[crossfade_samples:]
    ], dim=0)

    return result


def apply_fade_in_out(audio: torch.Tensor, fade_in_samples: int = 0, fade_out_samples: int = 0) -> torch.Tensor:
    """
    Apply fade in and fade out to audio.

    Args:
        audio: Audio tensor (1D)
        fade_in_samples: Number of samples for fade in
        fade_out_samples: Number of samples for fade out

    Returns:
        Audio with fades applied
    """
    audio = audio.clone()

    if fade_in_samples > 0 and len(audio) > fade_in_samples:
        fade_in = torch.linspace(0.0, 1.0, fade_in_samples, device=audio.device, dtype=audio.dtype)
        audio[:fade_in_samples] = audio[:fade_in_samples] * fade_in

    if fade_out_samples > 0 and len(audio) > fade_out_samples:
        fade_out = torch.linspace(1.0, 0.0, fade_out_samples, device=audio.device, dtype=audio.dtype)
        audio[-fade_out_samples:] = audio[-fade_out_samples:] * fade_out

    return audio

class MiraTTS:

    def __init__(self, model_dir="YatharthS/MiraTTS", tp=1, enable_prefix_caching=True, cache_max_entry_count=0.2):
        
        backend_config = TurbomindEngineConfig(cache_max_entry_count=cache_max_entry_count, tp=tp, dtype='bfloat16', enable_prefix_caching=enable_prefix_caching)
        self.pipe = pipeline(model_dir, backend_config=backend_config)
        self.gen_config = GenerationConfig(top_p=0.95,
                              top_k=50,
                              temperature=0.8,
                              max_new_tokens=1024,
                              repetition_penalty=1.2,
                              do_sample=True,
                              min_p=0.05)
        self.codec = TTSCodec()

    def set_params(self, top_p=0.95, top_k=50, temperature=0.8, max_new_tokens=1024, repetition_penalty=1.2, min_p=0.05):
        """sets sampling parameters for the llm"""
      
        self.gen_config = GenerationConfig(top_p=top_p, top_k=top_k, temperature=temperature, max_new_tokens=max_new_tokens, repetition_penalty=repetition_penalty, min_p=min_p, do_sample=True)
      
    def c_cache(self):
        clear_cache()

    def split_text(self, text):
        return split_text(text)
        
    def encode_audio(self, audio_file):
        """encodes audio into context tokens"""
      
        context_tokens = self.codec.encode(audio_file)
        return context_tokens

        
    def generate(self, text, context_tokens, fade_in_ms: int = 10, fade_out_ms: int = 50):
        """
        Generates speech from input text.

        Args:
            text: Input text
            context_tokens: Context tokens from reference audio
            fade_in_ms: Fade in duration in milliseconds
            fade_out_ms: Fade out duration in milliseconds
        """
        formatted_prompt = self.codec.format_prompt(text, context_tokens, None)

        response = self.pipe([formatted_prompt], gen_config=self.gen_config, do_preprocess=False)
        audio = self.codec.decode(response[0].text, context_tokens)

        # Apply fade in/out
        sample_rate = 48000
        fade_in_samples = int(fade_in_ms * sample_rate / 1000)
        fade_out_samples = int(fade_out_ms * sample_rate / 1000)
        audio = apply_fade_in_out(audio, fade_in_samples, fade_out_samples)

        return audio

    def batch_generate(self, prompts, context_tokens, crossfade_ms: int = 50, fade_in_ms: int = 10, fade_out_ms: int = 50):
        """
        Generates speech from text, for larger batch size with crossfade.

        Args:
            prompts (list): Input for tts model, list of prompts
            context_tokens (list): List of context tokens
            crossfade_ms: Crossfade duration between sentences in milliseconds
            fade_in_ms: Fade in duration at the beginning in milliseconds
            fade_out_ms: Fade out duration at the end in milliseconds
        """
        formatted_prompts = []
        for prompt, context_token in zip(prompts, cycle(context_tokens)):
            formatted_prompt = self.codec.format_prompt(prompt, context_token, None)
            formatted_prompts.append(formatted_prompt)

        responses = self.pipe(formatted_prompts, gen_config=self.gen_config, do_preprocess=False)
        generated_tokens = [response.text for response in responses]

        audios = []
        for generated_token, context_token in zip(generated_tokens, cycle(context_tokens)):
            audio = self.codec.decode(generated_token, context_token)
            audios.append(audio)

        # Apply crossfade between audio segments
        sample_rate = 48000
        crossfade_samples = int(crossfade_ms * sample_rate / 1000)

        if len(audios) == 1:
            combined_audio = audios[0]
        else:
            # Start with first audio
            combined_audio = audios[0]
            # Crossfade each subsequent audio
            for audio in audios[1:]:
                combined_audio = crossfade_audio(combined_audio, audio, crossfade_samples)

        # Apply fade in at the beginning and fade out at the end
        fade_in_samples = int(fade_in_ms * sample_rate / 1000)
        fade_out_samples = int(fade_out_ms * sample_rate / 1000)
        combined_audio = apply_fade_in_out(combined_audio, fade_in_samples, fade_out_samples)

        return combined_audio
            

