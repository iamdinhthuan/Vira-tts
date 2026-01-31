import time
import os
import torch
import gradio as gr
from huggingface_hub import snapshot_download
from mira.model import MiraTTS
from mira.utils import split_text

# Model config
HF_MODEL_ID = "dolly-vn/Vira-TTS"
MODEL_PATH = "model_pretrained"

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
    else:
        print(f"✅ Model found at: {MODEL_PATH}")

# Download model if needed
download_model_if_needed()

print("🔄 Loading Vira-TTS...")
mira_tts = MiraTTS(MODEL_PATH)
print("✅ Model loaded!")

SAMPLE_RATE = 48000


def generate_speech(text: str, reference_audio: str):
    """Generate speech from text using reference audio for voice cloning."""

    if not text.strip():
        return None, "Vui lòng nhập văn bản."

    if reference_audio is None:
        return None, "Vui lòng upload file audio tham chiếu."

    try:
        # Encode reference audio
        context_tokens = mira_tts.encode_audio(reference_audio)

        # Split text into sentences
        sentences = split_text(text)

        # Generate audio and measure time
        start_time = time.time()

        if len(sentences) == 1:
            # Single sentence - use generate
            audio = mira_tts.generate(sentences[0], context_tokens)
        else:
            # Multiple sentences - use batch_generate
            audio = mira_tts.batch_generate(sentences, [context_tokens])

        inference_time = time.time() - start_time

        # Calculate RTF
        audio_np = audio.float().cpu().numpy()
        audio_duration = len(audio_np) / SAMPLE_RATE
        rtf = inference_time / audio_duration

        # Create stats message
        stats = f"📝 Số câu: {len(sentences)} | ⏱️ Inference: {inference_time:.2f}s | 🎵 Audio: {audio_duration:.2f}s | 📊 RTF: {rtf:.4f}"

        return (SAMPLE_RATE, audio_np), stats

    except Exception as e:
        return None, f"Lỗi: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="Vira-TTS Vietnamese", theme=gr.themes.Soft()) as demo: # type: ignore
    gr.Markdown("""
    # 🎙️ Vira-TTS Vietnamese
    ### Text-to-Speech với Voice Cloning

    Upload một file audio tham chiếu để clone giọng nói, sau đó nhập văn bản để tạo audio.

    **Upsampler:** FlashSR (48kHz output, 14x realtime)
    """)

    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="Văn bản",
                placeholder="Nhập văn bản tiếng Việt tại đây...",
                lines=5
            )
            reference_audio = gr.Audio(
                label="Audio tham chiếu (để clone giọng)",
                type="filepath"
            )
            generate_btn = gr.Button("🎵 Tạo Audio", variant="primary", size="lg")

        with gr.Column(scale=1):
            output_audio = gr.Audio(
                label="Audio đầu ra",
                type="numpy"
            )
            stats_output = gr.Textbox(
                label="Thống kê",
                interactive=False
            )

    # Example texts (click to fill)
    gr.Markdown("### 📝 Ví dụ văn bản:")
    with gr.Row():
        gr.Button("Xin chào, tôi là trợ lý ảo.").click(
            fn=lambda: "Xin chào, tôi là trợ lý ảo Vira-TTS.",
            outputs=[text_input]
        )
        gr.Button("Thời tiết hôm nay").click(
            fn=lambda: "Hôm nay thời tiết rất đẹp, chúng ta đi dạo nhé!",
            outputs=[text_input]
        )
        gr.Button("Công nghệ AI").click(
            fn=lambda: "Công nghệ trí tuệ nhân tạo đang phát triển rất nhanh chóng.",
            outputs=[text_input]
        )

    # Event handler
    generate_btn.click(
        fn=generate_speech,
        inputs=[text_input, reference_audio],
        outputs=[output_audio, stats_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

