import time
import torch
import gradio as gr
from mira.model_novasr import MiraTTSNovaSR
from mira.utils import split_text

# Load model globally (load once at startup)
print("Loading MiraTTS with NovaSR upsampler...")
mira_tts = MiraTTSNovaSR('outputs_vi/checkpoint-25000')
print("Model loaded!")

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
with gr.Blocks(title="MiraTTS Vietnamese", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎙️ MiraTTS Vietnamese
    ### Text-to-Speech với Voice Cloning (NovaSR - 3600x realtime)

    Upload một file audio tham chiếu để clone giọng nói, sau đó nhập văn bản để tạo audio.
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
    
    # Examples
    gr.Examples(
        examples=[
            ["Xin chào, tôi là trợ lý ảo MiraTTS."],
            ["Hôm nay thời tiết rất đẹp, chúng ta đi dạo nhé!"],
            ["Công nghệ trí tuệ nhân tạo đang phát triển rất nhanh chóng."],
        ],
        inputs=[text_input]
    )
    
    # Event handler
    generate_btn.click(
        fn=generate_speech,
        inputs=[text_input, reference_audio],
        outputs=[output_audio, stats_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

