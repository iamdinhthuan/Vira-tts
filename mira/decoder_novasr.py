"""
Custom AudioDecoder using NovaSR instead of FlashSR for audio upsampling.
NovaSR is ~250x faster and ~20,000x smaller than FlashSR.
"""

import re
import torch
import numpy as np
import onnxruntime as ort
from NovaSR import FastSR
from huggingface_hub import snapshot_download


class AudioTokenizer:
    """Audio tokenizer using safetensors model."""
    
    def __init__(self, model_path):
        from safetensors.torch import load_file
        import torch.nn as nn
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the detokenizer model
        state_dict = load_file(model_path)
        
        # Infer model architecture from state dict
        self.model = self._build_model(state_dict)
        self.model.to(self.device)
        self.model.eval()
    
    def _build_model(self, state_dict):
        """Build model from state dict - this matches the original AudioTokenizer."""
        import torch.nn as nn
        
        # Simple pass-through for now - the actual model structure
        # depends on the original implementation
        class Detokenizer(nn.Module):
            def __init__(self, state_dict):
                super().__init__()
                # Register all parameters from state dict
                for name, tensor in state_dict.items():
                    name = name.replace('.', '_')
                    self.register_buffer(name, tensor)
            
            def forward(self, x):
                return x
        
        return Detokenizer(state_dict)
    
    def decode(self, x):
        """Decode tokens to audio."""
        with torch.no_grad():
            return self.model(x)


class AudioDecoderNovaSR:
    """
    AudioDecoder that uses NovaSR for upsampling instead of FlashSR.
    Drop-in replacement for ncodec.decoder.model.AudioDecoder
    """
    
    def __init__(self, decoder_paths=None):
        if decoder_paths is None:
            d_path = snapshot_download("YatharthS/MiraTTS")
            decoder_paths = f"{d_path}/decoders"
        
        sess_options = ort.SessionOptions()
        providers = [
            ("CUDAExecutionProvider", {"device_id": 0}),
            ("CPUExecutionProvider", {})
        ]
        
        self.processor_detokenizer = ort.InferenceSession(
            f"{decoder_paths}/processer.onnx", 
            sess_options, 
            providers=providers
        )
        
        # Import original AudioTokenizer from ncodec
        from ncodec.decoder.model_utils import AudioTokenizer
        self.audio_detokenizer = AudioTokenizer(f'{decoder_paths}/detokenizer.safetensors')
        
        # Use NovaSR instead of FlashSR
        print("Loading NovaSR upsampler (50KB, 3600x realtime)...")
        self.upsampler = FastSR(half=True)
        print("NovaSR loaded!")
    
    @torch.inference_mode()
    def detokenize(self, context_tokens, speech_tokens):
        """Detokenize speech tokens to audio waveform."""
        
        # Parse speech tokens
        speech_tokens = (
            torch.tensor([int(token) for token in re.findall(r"speech_token_(\d+)", speech_tokens)])
            .long()
            .unsqueeze(0)
        ).numpy()
        
        # Parse context tokens
        context_tokens = (
            torch.tensor([int(token) for token in re.findall(r"context_token_(\d+)", context_tokens)])
            .long()
            .unsqueeze(0).unsqueeze(0)
        ).numpy().astype(np.int32)
        
        # Process through ONNX model
        x = self.processor_detokenizer.run(
            ["preprocessed_output"], 
            {"context_tokens": context_tokens, "speech_tokens": speech_tokens}
        )
        x = torch.from_numpy(x[0]).to("cuda:0")
        
        # Decode to low-res audio
        lowres_wav = self.audio_detokenizer.decode(x).squeeze(0)
        
        # Upsample with NovaSR
        # NovaSR expects shape: (batch, 1, samples)
        if lowres_wav.dim() == 1:
            lowres_wav = lowres_wav.unsqueeze(0).unsqueeze(0)
        elif lowres_wav.dim() == 2:
            lowres_wav = lowres_wav.unsqueeze(1)
        
        lowres_wav = lowres_wav.half()
        highres_wav = self.upsampler.infer(lowres_wav)
        
        return highres_wav.squeeze(0)

