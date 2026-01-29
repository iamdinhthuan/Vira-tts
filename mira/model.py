import gc
import torch
from itertools import cycle
from ncodec.codec import TTSCodec
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig

from mira.utils import clear_cache, split_text

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

        
    def generate(self, text, context_tokens):
        """generates speech from input text"""
        formatted_prompt = self.codec.format_prompt(text, context_tokens, None)
      
        response = self.pipe([formatted_prompt], gen_config=self.gen_config, do_preprocess=False)
        audio = self.codec.decode(response[0].text, context_tokens)
        return audio
      
    def batch_generate(self, prompts, context_tokens):
        """
        Generates speech from text, for larger batch size

        Args:
            prompt (list): Input for tts model, list of prompts
            voice (list): Description of voice, list of voices respective to prompt
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
        audios = torch.cat(audios, dim=0)
      
        return audios
            

