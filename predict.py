import time
import soundfile as sf
from mira.model import MiraTTS

# Load finetuned Vietnamese checkpoint
mira_tts = MiraTTS('outputs_vi/checkpoint-25000')  ## Load local finetuned model

file = "2.wav"  ## can be mp3/wav/ogg or anything that librosa supports
text = "Xin chào, đây là giọng nói tiếng Việt được tạo bởi mô hình MiraTTS đã finetune."

context_tokens = mira_tts.encode_audio(file)

# Generate audio and measure time
start_time = time.time()
audio = mira_tts.generate(text, context_tokens)
inference_time = time.time() - start_time

# Calculate RTF (Real-Time Factor)
sample_rate = 48000
audio_duration = len(audio) / sample_rate
rtf = inference_time / audio_duration

print(f"Inference time: {inference_time:.2f}s")
print(f"Audio duration: {audio_duration:.2f}s")
print(f"RTF: {rtf:.4f} (lower is better, <1 means faster than real-time)")

# Save audio (convert to float32 for soundfile compatibility)
output_path = "output.wav"
sf.write(output_path, audio.float().cpu().numpy(), sample_rate)
print(f"Audio saved to: {output_path}")