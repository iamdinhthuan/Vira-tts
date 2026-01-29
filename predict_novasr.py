"""
Test script for MiraTTS with NovaSR upsampler.
NovaSR: 50KB model, 3600x realtime speed (vs FlashSR: 1GB, 14x realtime)
"""

import time
import soundfile as sf
from mira.model_novasr import MiraTTSNovaSR

# Load finetuned Vietnamese checkpoint with NovaSR
print("Loading MiraTTS with NovaSR upsampler...")
mira_tts = MiraTTSNovaSR('outputs_vi/checkpoint-25000')
print("Model loaded!")

file = "2.wav"  # Reference audio file
text = "Xin chào, đây là giọng nói tiếng Việt được tạo bởi mô hình MiraTTS với NovaSR."

context_tokens = mira_tts.encode_audio(file)

# Generate audio and measure time
start_time = time.time()
audio = mira_tts.generate(text, context_tokens)
inference_time = time.time() - start_time

# Calculate RTF (Real-Time Factor)
sample_rate = 48000
audio_np = audio.float().cpu().numpy()
audio_duration = len(audio_np) / sample_rate
rtf = inference_time / audio_duration

print(f"Inference time: {inference_time:.2f}s")
print(f"Audio duration: {audio_duration:.2f}s")
print(f"RTF: {rtf:.4f} (lower is better, <1 means faster than real-time)")

# Save audio
output_path = "output_novasr.wav"
sf.write(output_path, audio_np, sample_rate)
print(f"Audio saved to: {output_path}")

