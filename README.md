# Vira-TTS

Vietnamese Text-to-Speech với Voice Cloning, được finetune từ [MiraTTS](https://huggingface.co/YatharthS/MiraTTS) cho tiếng Việt.

Sử dụng [NovaSR](https://github.com/ysharma3501/NovaSR) (50KB, 3600x realtime) thay vì FlashSR để upscale audio lên 48kHz.

## ✨ Tính năng

- 🇻🇳 **Tiếng Việt**: Đã finetune cho tiếng Việt
- 🎙️ **Voice Cloning**: Clone giọng nói từ audio tham chiếu
- ⚡ **Siêu nhanh**: NovaSR upsampling 3600x realtime
- 🎵 **Chất lượng cao**: Audio 48kHz rõ ràng
- 💾 **Nhẹ**: NovaSR chỉ 50KB (so với FlashSR)

## 📦 Cài đặt

```bash
pip install git+https://github.com/iamdinhthuan/Vira-tts.git
```

Hoặc cài thủ công:
```bash
git clone https://github.com/iamdinhthuan/Vira-tts.git
cd Vira-tts
pip install -e .
pip install git+https://github.com/ysharma3501/NovaSR.git
```

## 🚀 Sử dụng

### Inference cơ bản

```python
from mira.model_novasr import MiraTTSNovaSR

# Load model (thay bằng checkpoint của bạn)
mira_tts = MiraTTSNovaSR('outputs_vi/checkpoint-25000')

# Audio tham chiếu để clone giọng
file = "reference.wav"
text = "Xin chào, đây là giọng nói tiếng Việt."

context_tokens = mira_tts.encode_audio(file)
audio = mira_tts.generate(text, context_tokens)

# Lưu audio
import soundfile as sf
sf.write("output.wav", audio.float().cpu().numpy(), 48000)
```

### Batch inference (nhiều câu)

```python
from mira.model_novasr import MiraTTSNovaSR

mira_tts = MiraTTSNovaSR('outputs_vi/checkpoint-25000')

file = "reference.wav"
texts = [
    "Xin chào, tôi là trợ lý ảo.",
    "Hôm nay thời tiết rất đẹp.",
    "Công nghệ AI đang phát triển nhanh chóng."
]

context_tokens = [mira_tts.encode_audio(file)]
audio = mira_tts.batch_generate(texts, context_tokens)
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
│   ├── model.py          # MiraTTS gốc (FlashSR)
│   ├── model_novasr.py   # MiraTTS với NovaSR ⭐
│   ├── codec_novasr.py   # Codec wrapper cho NovaSR
│   ├── decoder_novasr.py # Decoder với NovaSR
│   └── utils.py          # Utilities (split_text, punc_norm)
├── app.py                # Gradio Web UI
├── predict.py            # Script test FlashSR
├── predict_novasr.py     # Script test NovaSR ⭐
└── outputs_vi/           # Checkpoint finetune tiếng Việt
```

## 🔧 So sánh Upsampler

| Model | Speed | Size | Chất lượng |
|-------|-------|------|------------|
| FlashSR | 14x realtime | ~1GB | Tốt |
| **NovaSR** | **3600x realtime** | **~50KB** | **Tương đương** |

## 🙏 Credits

- [MiraTTS](https://github.com/ysharma3501/MiraTTS) - Model gốc
- [Spark-TTS](https://huggingface.co/SparkAudio/Spark-TTS-0.5B) - Base model
- [NovaSR](https://github.com/ysharma3501/NovaSR) - Audio super-resolution
- [LMDeploy](https://github.com/InternLM/lmdeploy) - LLM inference optimization

## 📧 Liên hệ

GitHub: [@iamdinhthuan](https://github.com/iamdinhthuan)
