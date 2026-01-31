#!/usr/bin/env python3
"""
Vira-TTS CLI Inference

Usage:
    python infer.py --text "Xin chào, tôi là Vira." --reference audio.wav --output output.wav
    python infer.py --text "Năm 2024 là năm tuyệt vời." --reference audio.wav  # auto save to output.wav
    python infer.py --text-file input.txt --reference audio.wav --output output.wav
"""

import os
import argparse
import time
import soundfile as sf
from huggingface_hub import snapshot_download


# Model config
HF_MODEL_ID = "dolly-vn/Vira-TTS"
MODEL_PATH = "model_pretrained"
SAMPLE_RATE = 48000


def download_model_if_needed():
    """Download model from HuggingFace if not exists locally."""
    if not os.path.exists(MODEL_PATH) or not os.listdir(MODEL_PATH):
        print(f"📥 Downloading model from HuggingFace: {HF_MODEL_ID}...")
        snapshot_download(
            repo_id=HF_MODEL_ID,
            local_dir=MODEL_PATH,
            local_dir_use_symlinks=False
        )
        print("✅ Model downloaded!")


def main():
    parser = argparse.ArgumentParser(
        description="Vira-TTS: Vietnamese Text-to-Speech CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python infer.py --text "Xin chào" --reference ref.wav
  python infer.py --text "Năm 2024" --reference ref.wav --output out.wav
  python infer.py --text-file story.txt --reference ref.wav --output story.wav
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", "-t", type=str, help="Text to synthesize")
    input_group.add_argument("--text-file", "-f", type=str, help="Text file to read from")
    
    # Required
    parser.add_argument("--reference", "-r", type=str, required=True,
                        help="Reference audio file for voice cloning")
    
    # Optional
    parser.add_argument("--output", "-o", type=str, default="output.wav",
                        help="Output audio file (default: output.wav)")
    parser.add_argument("--model", "-m", type=str, default=MODEL_PATH,
                        help=f"Model path (default: {MODEL_PATH})")
    parser.add_argument("--no-normalize", action="store_true",
                        help="Disable Vietnamese text normalization")
    
    args = parser.parse_args()
    
    # Get text
    if args.text:
        text = args.text
    else:
        with open(args.text_file, "r", encoding="utf-8") as f:
            text = f.read().strip()
    
    if not text:
        print("❌ Error: Empty text input")
        return 1
    
    # Check reference audio
    if not os.path.exists(args.reference):
        print(f"❌ Error: Reference audio not found: {args.reference}")
        return 1
    
    # Download model if needed
    download_model_if_needed()
    
    # Import after download check (avoid slow imports before args validation)
    from mira.model import MiraTTS
    from mira.utils import split_text
    
    # Load model
    print(f"🔄 Loading model from: {args.model}")
    mira_tts = MiraTTS(args.model)
    print("✅ Model loaded!")
    
    # Encode reference audio
    print(f"🎤 Encoding reference audio: {args.reference}")
    context_tokens = mira_tts.encode_audio(args.reference)
    
    # Split and normalize text
    print(f"📝 Input text: {text[:100]}{'...' if len(text) > 100 else ''}")
    sentences = split_text(text, normalize=not args.no_normalize)
    print(f"📝 Sentences: {len(sentences)}")
    
    # Generate audio
    print("🎵 Generating audio...")
    start_time = time.time()
    
    if len(sentences) == 1:
        audio = mira_tts.generate(sentences[0], context_tokens)
    else:
        audio = mira_tts.batch_generate(sentences, [context_tokens])
    
    inference_time = time.time() - start_time
    
    # Save audio
    audio_np = audio.float().cpu().numpy()
    audio_duration = len(audio_np) / SAMPLE_RATE
    rtf = inference_time / audio_duration
    
    sf.write(args.output, audio_np, SAMPLE_RATE)
    
    # Print stats
    print()
    print("=" * 50)
    print(f"✅ Audio saved to: {args.output}")
    print(f"📊 Duration: {audio_duration:.2f}s")
    print(f"⏱️  Inference time: {inference_time:.2f}s")
    print(f"📈 RTF: {rtf:.4f}")
    print("=" * 50)
    
    return 0


if __name__ == "__main__":
    exit(main())

