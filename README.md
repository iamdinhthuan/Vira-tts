# Vira-TTS

Vietnamese Text-to-Speech with Voice Cloning, finetuned from [MiraTTS](https://huggingface.co/YatharthS/MiraTTS) for Vietnamese.

**Model:** [dolly-vn/Vira-TTS](https://huggingface.co/dolly-vn/Vira-TTS)

## Features

- **Vietnamese TTS**: Finetuned specifically for Vietnamese language
- **Voice Cloning**: Clone any voice from a reference audio sample
- **High Quality**: 48kHz audio output using [FlashSR](https://github.com/ysharma3501/FlashSR) upsampling (14x realtime)
- **Text Normalization**: Automatic conversion of numbers, abbreviations to spoken form using [soe-vinorm](https://github.com/vinhdq842/soe-vinorm.git)
- **Smooth Audio**: Crossfade support for seamless sentence transitions

## Installation

```bash
pip install git+https://github.com/iamdinhthuan/Vira-tts.git
```

Or install manually:

```bash
git clone https://github.com/iamdinhthuan/Vira-tts.git
cd Vira-tts
pip install -e .
```

## Quick Start

The model will be automatically downloaded from HuggingFace on first run.

### Basic Inference

```python
from mira.model import MiraTTS

mira_tts = MiraTTS('model_pretrained')

# Reference audio for voice cloning
reference_audio = "reference.wav"
text = "Xin chào, đây là giọng nói tiếng Việt."

context_tokens = mira_tts.encode_audio(reference_audio)
audio = mira_tts.generate(text, context_tokens)

# Save audio
import soundfile as sf
sf.write("output.wav", audio.float().cpu().numpy(), 48000)
```

### Text Normalization

Vira-TTS automatically normalizes Vietnamese text (numbers, abbreviations to spoken form):

```python
from mira.utils import split_text, normalize_vietnamese

text = "Từ năm 2021 đến nay, đây là lần thứ 3."
normalized = normalize_vietnamese(text)
# Output: "Từ năm hai nghìn không trăm hai mươi mốt đến nay, đây là lần thứ ba."

# split_text() applies normalization automatically
sentences = split_text(text)
```

### Batch Inference with Crossfade

For multiple sentences with smooth transitions:

```python
from mira.model import MiraTTS

mira_tts = MiraTTS('model_pretrained')

texts = [
    "Xin chào, tôi là trợ lý ảo.",
    "Hôm nay thời tiết rất đẹp.",
    "Công nghệ AI đang phát triển nhanh chóng."
]

context_tokens = [mira_tts.encode_audio("reference.wav")]

audio = mira_tts.batch_generate(
    texts,
    context_tokens,
    crossfade_ms=50,   # 50ms crossfade between sentences
    fade_in_ms=10,     # 10ms fade in at start
    fade_out_ms=50     # 50ms fade out at end
)
```

### Web UI

```bash
python app.py
```

Open browser at `http://localhost:7860`. Model will be downloaded automatically on first run.

### Docker (GPU)

Requires NVIDIA Container Toolkit.

```bash
docker build -t vira-tts:latest .
docker run --rm -it --gpus all -p 7860:7860 \
  -v $(pwd)/.hf-cache:/app/.cache/huggingface \
  vira-tts:latest
```

Model sẽ tự tải ở lần chạy đầu. Mount cache để tránh tải lại.

### CLI

```bash
# Basic usage
python infer.py --text "Xin chào, tôi là Vira." --reference audio.wav

# Specify output file
python infer.py --text "Năm 2024 là năm tuyệt vời." --reference audio.wav --output output.wav

# Read from text file
python infer.py --text-file story.txt --reference audio.wav --output story.wav

# Disable text normalization
python infer.py --text "Xin chào" --reference audio.wav --no-normalize
```

## Project Structure

```
Vira-tts/
├── mira/
│   ├── model.py          # MiraTTS with FlashSR and crossfade
│   └── utils.py          # Text processing utilities
├── app.py                # Gradio Web UI
├── infer.py              # CLI inference script
├── predict.py            # Test script
└── model_pretrained/     # Model files (auto-downloaded)
```

## Audio Processing

**Crossfade**: When joining multiple sentences, crossfade is applied to avoid audio clicks at boundaries (default: 50ms).

**Fade in/out**: Smooth fade applied at the beginning (10ms) and end (50ms) of the final audio.

## Credits

- [MiraTTS](https://github.com/ysharma3501/MiraTTS) - Original model
- [Spark-TTS](https://huggingface.co/SparkAudio/Spark-TTS-0.5B) - Base model
- [FlashSR](https://github.com/ysharma3501/FlashSR) - Audio super-resolution
- [LMDeploy](https://github.com/InternLM/lmdeploy) - LLM inference optimization
- [soe-vinorm](https://github.com/vinhdq842/soe-vinorm) - Vietnamese text normalization

## License

MIT

## Contact

GitHub: [@iamdinhthuan](https://github.com/iamdinhthuan)
