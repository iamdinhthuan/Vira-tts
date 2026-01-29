"""
Custom AudioDecoder using NovaSR instead of FlashSR for audio upsampling.
NovaSR is ~250x faster and ~20,000x smaller than FlashSR.
Includes VAD-based silence trimming before upsampling.
"""

import re
import torch
import numpy as np
import torchaudio
import onnxruntime as ort
from NovaSR import FastSR
from huggingface_hub import snapshot_download

# Global VAD model cache
_vad_model = None
_get_timestamps = None


def load_vad_model():
    """Load Silero VAD model (cached globally)."""
    global _vad_model, _get_timestamps

    if _vad_model is None:
        try:
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                trust_repo=True
            )
            _vad_model = model
            _get_timestamps = utils[0]  # get_speech_timestamps
            print("Silero VAD loaded!")
        except Exception as e:
            print(f"Failed to load VAD model: {e}")
            return None, None

    return _vad_model, _get_timestamps


def trim_silence_with_vad(audio_tensor: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
    """
    Trims silence/noise from the end of the audio using Silero VAD.

    Args:
        audio_tensor: Audio tensor at 16kHz
        sample_rate: Sample rate (should be 16000)

    Returns:
        Trimmed audio tensor
    """
    vad_model, get_timestamps = load_vad_model()
    if vad_model is None:
        return audio_tensor

    VAD_SR = 16000

    # Ensure tensor is float and 1D
    if audio_tensor.dim() > 1:
        audio_tensor = audio_tensor.squeeze()
    audio_tensor = audio_tensor.float()

    # Resample for VAD if necessary
    if sample_rate != VAD_SR:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=VAD_SR)
        vad_input = resampler(audio_tensor)
    else:
        vad_input = audio_tensor

    try:
        # Get speech timestamps
        speech_timestamps = get_timestamps(vad_input, vad_model, sampling_rate=VAD_SR)

        if not speech_timestamps:
            return audio_tensor

        # Get the end of the last speech chunk
        last_speech_end_vad = speech_timestamps[-1]['end']

        # Scale back to original sample rate
        scale_factor = sample_rate / VAD_SR
        cut_point = int(last_speech_end_vad * scale_factor)

        # Add small padding (50ms) to avoid cutting speech
        padding = int(0.05 * sample_rate)
        cut_point = min(cut_point + padding, audio_tensor.shape[0])

        trimmed_wav = audio_tensor[:cut_point]

        return trimmed_wav

    except Exception as e:
        print(f"VAD trimming failed: {e}")
        return audio_tensor


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
        """Detokenize speech tokens to audio waveform with VAD trimming."""

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

        # Decode to low-res audio (16kHz)
        lowres_wav = self.audio_detokenizer.decode(x).squeeze(0)

        # Trim silence with VAD before upsampling (audio is at 16kHz)
        lowres_wav_trimmed = trim_silence_with_vad(lowres_wav, sample_rate=16000)

        # Upsample with NovaSR
        # NovaSR expects shape: (batch, 1, samples)
        if lowres_wav_trimmed.dim() == 1:
            lowres_wav_trimmed = lowres_wav_trimmed.unsqueeze(0).unsqueeze(0)
        elif lowres_wav_trimmed.dim() == 2:
            lowres_wav_trimmed = lowres_wav_trimmed.unsqueeze(1)

        # Ensure on GPU and half precision
        lowres_wav_trimmed = lowres_wav_trimmed.to("cuda:0").half()
        highres_wav = self.upsampler.infer(lowres_wav_trimmed)

        return highres_wav.squeeze(0)

