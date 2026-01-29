# Vira-TTS

Vietnamese Text-to-Speech với Voice Cloning, được finetune từ [MiraTTS](https://huggingface.co/YatharthS/MiraTTS) cho tiếng Việt.

Sử dụng [FlashSR](https://github.com/ysharma3501/FlashSR) để upscale audio lên 48kHz chất lượng cao.

## ✨ Tính năng

- 🇻🇳 **Tiếng Việt**: Đã finetune cho tiếng Việt
- 🎙️ **Voice Cloning**: Clone giọng nói từ audio tham chiếu
- ⚡ **Nhanh**: FlashSR upsampling 14x realtime
- 🎵 **Chất lượng cao**: Audio 48kHz rõ ràng
- � **Crossfade**: Nối nhiều câu mượt mà với crossfade
- 📝 **Text Processing**: Chuẩn hóa dấu câu và cắt câu tự động

## 📦 Cài đặt

```bash
pip install git+https://github.com/iamdinhthuan/Vira-tts.git
```

Hoặc cài thủ công:
```bash
git clone https://github.com/iamdinhthuan/Vira-tts.git
cd Vira-tts
pip install -e .
```

## 🚀 Sử dụng

### Inference cơ bản

```python
from mira.model import MiraTTS

# Load model (thay bằng checkpoint của bạn)
mira_tts = MiraTTS('outputs_vi/checkpoint-25000')

# Audio tham chiếu để clone giọng
file = "reference.wav"
text = "Xin chào, đây là giọng nói tiếng Việt."

context_tokens = mira_tts.encode_audio(file)
audio = mira_tts.generate(text, context_tokens)

# Lưu audio
import soundfile as sf
sf.write("output.wav", audio.float().cpu().numpy(), 48000)
```

### Batch inference (nhiều câu với crossfade)

```python
from mira.model import MiraTTS

mira_tts = MiraTTS('outputs_vi/checkpoint-25000')

file = "reference.wav"
texts = [
    "Xin chào, tôi là trợ lý ảo.",
    "Hôm nay thời tiết rất đẹp.",
    "Công nghệ AI đang phát triển nhanh chóng."
]

context_tokens = [mira_tts.encode_audio(file)]

# Crossfade 50ms giữa các câu, fade in 10ms, fade out 50ms
audio = mira_tts.batch_generate(
    texts,
    context_tokens,
    crossfade_ms=50,
    fade_in_ms=10,
    fade_out_ms=50
)
```

### Gradio Web UI

```bash
python app.py
```

Mở trình duyệt tại `http://localhost:7860`

## 📁 Cấu trúc

```
Vira-tts/
├── mira/
│   ├── model.py          # MiraTTS với FlashSR và crossfade
│   └── utils.py          # Utilities (split_text, punc_norm)
├── app.py                # Gradio Web UI
├── predict.py            # Script test
└── outputs_vi/           # Checkpoint finetune tiếng Việt
```

## 🔧 Tính năng Audio

### Crossfade
Khi nối nhiều câu, sử dụng crossfade để tránh tiếng "click":
```
Câu 1: ───────────╲
                   ╳  ← Crossfade 50ms
Câu 2:            ╱───────────
```

### Fade in/out
Áp dụng fade ở đầu và cuối audio:
```
Audio: ╱─────────────────────╲
       ↑                     ↑
   Fade in 10ms         Fade out 50ms
```

## 🙏 Credits

- [MiraTTS](https://github.com/ysharma3501/MiraTTS) - Model gốc
- [Spark-TTS](https://huggingface.co/SparkAudio/Spark-TTS-0.5B) - Base model
- [FlashSR](https://github.com/ysharma3501/FlashSR) - Audio super-resolution
- [LMDeploy](https://github.com/InternLM/lmdeploy) - LLM inference optimization

## 📧 Liên hệ

GitHub: [@iamdinhthuan](https://github.com/iamdinhthuan)
