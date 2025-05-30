from huggingface_hub import snapshot_download
import gradio as gr
import openvino_genai
import librosa
import numpy as np
from threading import Lock, Event
from scipy.ndimage import uniform_filter1d
from queue import Queue, Empty
from googleapiclient.discovery import build
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import cpuinfo
import gc
import os

# Set CPU affinity for optimization
os.environ["GOMP_CPU_AFFINITY"] = "0-7"  # Use first 8 CPU cores
os.environ["OMP_NUM_THREADS"] = "8"

# Configuration constants
GOOGLE_API_KEY = "AIzaSyAo-1iW5MEZbc53DlEldtnUnDaYuTHUDH4"
GOOGLE_CSE_ID = "3027bedf3c88a4efb"
DEFAULT_MAX_TOKENS = 100
DEFAULT_NUM_IMAGES = 1
MAX_HISTORY_TURNS = 2
MAX_TOKENS_LIMIT = 1000

# Download models
start_time = time.time()
snapshot_download(repo_id="OpenVINO/mistral-7b-instruct-v0.1-int8-ov", local_dir="mistral-ov")
snapshot_download(repo_id="OpenVINO/whisper-tiny-fp16-ov", local_dir="whisper-ov-model")
print(f"Model download time: {time.time() - start_time:.2f} seconds")

# CPU-specific configuration
cpu_features = cpuinfo.get_cpu_info()['flags']
config_options = {}
if 'avx512' in cpu_features:
    config_options["ENFORCE_BF16"] = "YES"
    print("Using AVX512 optimizations")
elif 'avx2' in cpu_features:
    config_options["INFERENCE_PRECISION_HINT"] = "f32"
    print("Using AVX2 optimizations")

# Initialize models with performance flags
start_time = time.time()
mistral_pipe = openvino_genai.LLMPipeline(
    "mistral-ov", 
    device="CPU",
    config={
        "PERFORMANCE_HINT": "THROUGHPUT",
        **config_options
    }
)

whisper_pipe = openvino_genai.WhisperPipeline(
    "whisper-ov-model", 
    device="CPU"
)
pipe_lock = Lock()
print(f"Model initialization time: {time.time() - start_time:.2f} seconds")

# Warm up models
print("Warming up models...")
start_time = time.time()
with pipe_lock:
    mistral_pipe.generate("Warmup", openvino_genai.GenerationConfig(max_new_tokens=10))
    whisper_pipe.generate(np.zeros(16000, dtype=np.float32))
print(f"Model warmup time: {time.time() - start_time:.2f} seconds")

# Thread pools
generation_executor = ThreadPoolExecutor(max_workers=4)  # Increased workers
image_executor = ThreadPoolExecutor(max_workers=8)

def fetch_images(query: str, num: int = DEFAULT_NUM_IMAGES) -> list:
    """Fetch unique images by requesting different result pages"""
    start_time = time.time()
    
    if num <= 0:
        return []

    try:
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        image_links = []
        seen_urls = set()  # To track unique URLs
        
        # Start from different positions to get unique images
        for start_index in range(1, num * 2, 2):  # Step by 2 to get different pages
            if len(image_links) >= num:
                break
                
            res = service.cse().list(
                q=query,
                cx=GOOGLE_CSE_ID,
                searchType="image",
                num=1,  # Get one result per request
                start=start_index  # Start at different positions
            ).execute()
            
            if "items" in res and res["items"]:
                item = res["items"][0]
                # Skip duplicates
                if item["link"] not in seen_urls:
                    image_links.append(item["link"])
                    seen_urls.add(item["link"])
        
        print(f"Unique image fetch time: {time.time() - start_time:.2f} seconds")
        return image_links[:num]  # Return only the requested number
    except Exception as e:
        print(f"Error in image fetching: {e}")
        return []

def process_audio(data, sr):
    start_time = time.time()
    data = librosa.to_mono(data.T) if data.ndim > 1 else data
    data = data.astype(np.float32)
    data /= np.max(np.abs(data))
    rms = librosa.feature.rms(y=data, frame_length=2048, hop_length=512)[0]
    smoothed_rms = uniform_filter1d(rms, size=5)
    speech_frames = np.where(smoothed_rms > 0.025)[0]
    if not speech_frames.size:
        print(f"Audio processing time: {time.time() - start_time:.2f} seconds")
        return None
    start = max(0, int(speech_frames[0] * 512 - 0.1 * sr))
    end = min(len(data), int((speech_frames[-1] + 1) * 512 + 0.1 * sr))
    print(f"Audio processing time: {time.time() - start_time:.2f} seconds")
    return data[start:end]

def transcribe(audio):
    start_time = time.time()
    if audio is None:
        print(f"Transcription time: {time.time() - start_time:.2f} seconds")
        return ""
    sr, data = audio
    processed = process_audio(data, sr)
    if processed is None or len(processed) < 1600:
        print(f"Transcription time: {time.time() - start_time:.2f} seconds")
        return ""
    if sr != 16000:
        processed = librosa.resample(processed, orig_sr=sr, target_sr=16000)
    result = whisper_pipe.generate(processed)
    print(f"Transcription time: {time.time() - start_time:.2f} seconds")
    return result

def stream_answer(message: str, max_tokens: int, include_images: bool) -> str:
    start_time = time.time()
    response_queue = Queue()
    completion_event = Event()
    error = [None]

    optimized_config = openvino_genai.GenerationConfig(
        max_new_tokens=max_tokens,
        num_beams=1,
        do_sample=False,
        temperature=1.0,
        top_p=0.9,
        top_k=30,
        streaming=True,
        streaming_interval=5  # Batch tokens in groups of 5
    )

    def callback(tokens):  # Now accepts multiple tokens
        response_queue.put("".join(tokens))
        return openvino_genai.StreamingStatus.RUNNING

    def generate():
        try:
            with pipe_lock:
                mistral_pipe.generate(message, optimized_config, callback)
        except Exception as e:
            error[0] = str(e)
        finally:
            completion_event.set()

    generation_executor.submit(generate)

    accumulated = []
    token_count = 0
    last_gc = time.time()
    
    while not completion_event.is_set() or not response_queue.empty():
        if error[0]:
            yield f"Error: {error[0]}"
            print(f"Stream answer time: {time.time() - start_time:.2f} seconds")
            return
            
        try:
            token_batch = response_queue.get_nowait()
            accumulated.append(token_batch)
            token_count += len(token_batch)
            
            # Periodic garbage collection
            if time.time() - last_gc > 2.0:  # Every 2 seconds
                gc.collect()
                last_gc = time.time()
            
            yield "".join(accumulated)
        except Empty:
            continue
            
    print(f"Generated {token_count} tokens in {time.time() - start_time:.2f} seconds "
          f"({token_count/(time.time() - start_time):.2f} tokens/sec)")
    yield "".join(accumulated)

def run_chat(message: str, history: list, include_images: bool, max_tokens: int, num_images: int):
    start_time = time.time()
    final_text = ""
    
    # Create a placeholder for the streaming response
    history.append((message, "", []))
    rendered_history = render_history(history)
    yield rendered_history, gr.update(value="", interactive=False)
    
    # Stream tokens and update chatbot in real-time
    for output in stream_answer(message, max_tokens, include_images):
        final_text = output
        # Update only the last response in history
        updated_history = history[:-1] + [(message, final_text, [])]
        rendered_history = render_history(updated_history)
        yield rendered_history, gr.update(value="", interactive=False)
    
    images = []
    if include_images:
        images = fetch_images(message, num_images)
    
    # Update history with final response and images
    history[-1] = (message, final_text, images)
    if len(history) > MAX_HISTORY_TURNS:
        history = history[-MAX_HISTORY_TURNS:]
    
    rendered_history = render_history(history)
    print(f"Total chat time: {time.time() - start_time:.2f} seconds")
    yield rendered_history, gr.update(value="", interactive=True)

def render_history(history):
    start_time = time.time()
    rendered = []
    for user_msg, bot_msg, image_links in history:
        text = bot_msg
        if image_links:
            images_html = "".join(
                f"<img src='{url}' class='chat-image' onclick='showImage(\"{url}\")' />"
                for url in image_links
            )
            text += f"<br><br><b>📸 Related Visuals:</b><br><div style='display: flex; flex-wrap: wrap;'>{images_html}</div>"
        rendered.append((user_msg, text))
  
    return rendered

with gr.Blocks(css="""
    .processing {
        animation: pulse 1.5s infinite;
        color: #4a5568;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        margin: 10px 0;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    .chat-image {
        cursor: pointer;
        transition: transform 0.2s;
        max-height: 100px;
        margin: 4px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .chat-image:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .modal {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0,0,0,0.8);
        display: none;
        z-index: 1000;
        cursor: zoom-out;
    }
    .modal-content {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        max-width: 90%;
        max-height: 90%;
        background: white;
        padding: 10px;
        border-radius: 12px;
    }
    .modal-img {
        width: auto;
        height: auto;
        max-width: 100%;
        max-height: 100%;
        border-radius: 8px;
    }
    .chat-container {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .slider-container {
        margin-top: 20px;
        padding: 15px;
        border-radius: 10px;
        background-color: #f8f9fa;
    }
    .slider-label {
        font-weight: bold;
        margin-bottom: 5px;
    }
    .system-info {
        background-color: #7B9BDB;
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
        border-left: 4px solid #1890ff;
    }
    .typing-indicator {
        display: inline-block;
        position: relative;
        width: 40px;
        height: 20px;
    }
    .typing-dot {
        display: inline-block;
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: #4a5568;
        position: absolute;
        animation: typing 1.4s infinite ease-in-out;
    }
    .typing-dot:nth-child(1) {
        left: 0;
        animation-delay: 0s;
    }
    .typing-dot:nth-child(2) {
        left: 12px;
        animation-delay: 0.2s;
    }
    .typing-dot:nth-child(3) {
        left: 24px;
        animation-delay: 0.4s;
    }
    @keyframes typing {
        0%, 60%, 100% { transform: translateY(0); }
        30% { transform: translateY(-5px); }
    }
""") as demo:
    gr.Markdown("# 🤖 EDU CHAT BY PHANINDRA REDDY K")
    
    # System info banner
    gr.HTML("""
    <div class="system-info">
        <strong>Performance Optimized for High-RAM Systems</strong>
        <ul>
            <li>Adaptive resource allocation based on request type</li>
        </ul>
    </div>
    """)

    modal_html = """
    <div class="modal" id="imageModal" onclick="this.style.display='none'">
        <div class="modal-content">
            <img class="modal-img" id="expandedImg">
        </div>
    </div>
    <script>
    function showImage(url) {
        document.getElementById('expandedImg').src = url;
        document.getElementById('imageModal').style.display = 'block';
    }
    </script>
    """
    gr.HTML(modal_html)

    state = gr.State([])

    with gr.Column(scale=2, elem_classes="chat-container"):
        chatbot = gr.Chatbot(label="Conversation", height=500, bubble_full_width=False)

    with gr.Column(scale=1):
        gr.Markdown("### 💬 Ask Your Question")

        with gr.Row():
            user_input = gr.Textbox(
                placeholder="Type your question here...",
                label="",
                container=False,
                elem_id="question-input"
            )
            include_images = gr.Checkbox(
                label="Include Visuals",
                value=True,
                container=False,
                elem_id="image-checkbox"
            )

        # Add the sliders container
        with gr.Column(elem_classes="slider-container"):
            gr.Markdown("### ⚙️ Generation Settings")

            with gr.Row():
                max_tokens = gr.Slider(
                    minimum=10,
                    maximum=MAX_TOKENS_LIMIT,  # Increased to 1000
                    value=DEFAULT_MAX_TOKENS,
                    step=10,
                    label="Response Length (Tokens)",
                    info=f"Max: {MAX_TOKENS_LIMIT} tokens (for detailed explanations)",
                    elem_classes="slider-label"
                )

            # Conditionally visible image slider row
            with gr.Row(visible=True) as image_slider_row:
                num_images = gr.Slider(
                    minimum=0,
                    maximum=5,
                    value=DEFAULT_NUM_IMAGES,
                    step=1,
                    label="Number of Images",
                    info="Set to 0 to disable images",
                    elem_classes="slider-label"
                )

        with gr.Row():
            submit_btn = gr.Button("Send Text", variant="primary")
            mic_btn = gr.Button("Transcribe Voice", variant="secondary")
            mic = gr.Audio(
                sources=["microphone"],
                type="numpy",
                label="Voice Input",
                show_label=False,
                elem_id="voice-input"
            )

        processing = gr.HTML("""
            <div id="processing" style="display: none;">
                <div class="processing">🔮 Processing your request...</div>
            </div>
        """)

    # Toggle image slider visibility based on checkbox
    def toggle_image_slider(include_visuals):
        return gr.update(visible=include_visuals)
    
    include_images.change(
        fn=toggle_image_slider,
        inputs=include_images,
        outputs=image_slider_row
    )

    def toggle_processing():
        return gr.update(visible=True), gr.update(interactive=False)

    def hide_processing():
        return gr.update(visible=False), gr.update(interactive=True)

    # Update the submit_btn click handler to include streaming
    submit_btn.click(
        fn=toggle_processing,
        outputs=[processing, submit_btn]
    ).then(
        fn=lambda: (gr.update(visible=True), gr.update(interactive=False)),
        outputs=[processing, submit_btn]
    ).then(
        fn=run_chat,
        inputs=[user_input, state, include_images, max_tokens, num_images],
        outputs=[chatbot, user_input]
    ).then(
        fn=lambda: (gr.update(visible=False), gr.update(interactive=True)),
        outputs=[processing, submit_btn]
    )

    # Voice transcription remains the same
    mic_btn.click(
        fn=toggle_processing,
        outputs=[processing, mic_btn]
    ).then(
        fn=transcribe,
        inputs=mic,
        outputs=user_input
    ).then(
        fn=hide_processing,
        outputs=[processing, mic_btn]
    )

if __name__ == "__main__":
    demo.launch(share=True, debug=True)