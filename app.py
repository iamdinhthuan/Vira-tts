import time
import torch
import gradio as gr
from mira.model_novasr import MiraTTSNovaSR
from mira.model import MiraTTS
from mira.utils import split_text

# Load models globally (load once at startup)
MODEL_PATH = 'outputs_vi/checkpoint-25000'

print("Loading MiraTTS with NovaSR upsampler...")
mira_tts_novasr = MiraTTSNovaSR(MODEL_PATH)
print("NovaSR model loaded!")

print("Loading MiraTTS with FlashSR upsampler...")
mira_tts_flashsr = MiraTTS(MODEL_PATH)
print("FlashSR model loaded!")

SAMPLE_RATE = 48000


def generate_speech(text: str, reference_audio: str, upsampler: str):
    """Generate speech from text using reference audio for voice cloning."""

    if not text.strip():
        return None, "Vui lòng nhập văn bản."

    if reference_audio is None:
        return None, "Vui lòng upload file audio tham chiếu."

    try:
        # Select model based on upsampler choice
        if upsampler == "NovaSR (50KB, 3600x realtime)":
            mira_tts = mira_tts_novasr
            upsampler_name = "NovaSR"
        else:
            mira_tts = mira_tts_flashsr
            upsampler_name = "FlashSR"

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
        stats = f"🔧 {upsampler_name} | 📝 Số câu: {len(sentences)} | ⏱️ Inference: {inference_time:.2f}s | 🎵 Audio: {audio_duration:.2f}s | 📊 RTF: {rtf:.4f}"

        return (SAMPLE_RATE, audio_np), stats

    except Exception as e:
        return None, f"Lỗi: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="Vira-TTS Vietnamese", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎙️ Vira-TTS Vietnamese
    ### Text-to-Speech với Voice Cloning

    Upload một file audio tham chiếu để clone giọng nói, sau đó nhập văn bản để tạo audio.

    | Upsampler | Speed | Size |
    |-----------|-------|------|
    | **NovaSR** | 3600x realtime | 50KB |
    | FlashSR | 14x realtime | 1GB |
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
            upsampler_choice = gr.Radio(
                label="Upsampler",
                choices=["NovaSR (50KB, 3600x realtime)", "FlashSR (1GB, 14x realtime)"],
                value="NovaSR (50KB, 3600x realtime)"
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
            fn=lambda: "Xin chào, tôi là trợ lý ảo MiraTTS.",
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
        inputs=[text_input, reference_audio, upsampler_choice],
        outputs=[output_audio, stats_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

