from huggingface_hub import snapshot_download

# Download models from Hugging Face to local folders
snapshot_download(
    repo_id="OpenVINO/Mistral-7B-Instruct-v0.2-int4-ov",
    local_dir="mistral-ov"
)
snapshot_download(
    repo_id="OpenVINO/whisper-tiny-fp16-ov",
    local_dir="whisper-ov-model"
)

import gradio as gr
import openvino_genai
import librosa
import numpy as np
from threading import Thread, Lock, Event
from scipy.ndimage import uniform_filter1d
from queue import Queue, Empty

# Initialize Mistral pipeline
mistral_pipe = openvino_genai.LLMPipeline("mistral-ov", device="CPU")
config = openvino_genai.GenerationConfig(
    max_new_tokens=100,
    num_beams=1,
    do_sample=False,
    temperature=0.0,
    top_p=1.0,
    top_k=50  
)
pipe_lock = Lock()

# Initialize Whisper pipeline 
whisper_pipe = openvino_genai.WhisperPipeline("whisper-ov-model", device="CPU")

def process_audio(data, sr):
    """Audio processing with silence trimming"""
    data = librosa.to_mono(data.T) if data.ndim > 1 else data
    data = data.astype(np.float32)
    data /= np.max(np.abs(data))
    
    # Voice activity detection
    frame_length, hop_length = 2048, 512
    rms = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)[0]
    smoothed_rms = uniform_filter1d(rms, size=5)
    speech_frames = np.where(smoothed_rms > 0.025)[0]
    
    if not speech_frames.size:
        return None
    
    start = max(0, int(speech_frames[0] * hop_length - 0.1*sr))
    end = min(len(data), int((speech_frames[-1]+1) * hop_length + 0.1*sr))
    return data[start:end]

def transcribe(audio):
    """Audio to text transcription"""
    sr, data = audio
    processed = process_audio(data, sr)
    if processed is None or len(processed) < 1600:
        return ""
    
    if sr != 16000:
        processed = librosa.resample(processed, orig_sr=sr, target_sr=16000)
    
    return whisper_pipe.generate(processed)

def stream_generator(message, history):
    response_queue = Queue()
    completion_event = Event()
    error_message = [None]

    def callback(token):
        response_queue.put(token)
        return openvino_genai.StreamingStatus.RUNNING

    def generate():
        try:
            with pipe_lock:
                mistral_pipe.generate(message, config, callback)
        except Exception as e:
            error_message[0] = str(e)
        finally:
            completion_event.set()

    Thread(target=generate, daemon=True).start()

    accumulated = []
    while not completion_event.is_set() or not response_queue.empty():
        if error_message[0]:
            yield f"Error: {error_message[0]}"
            return

        try:
            token = response_queue.get_nowait()
            accumulated.append(token)
            yield "".join(accumulated)
        except Empty:
            continue

    yield "".join(accumulated)

with gr.Blocks() as demo:
    chat_interface = gr.ChatInterface(
        stream_generator,
        textbox=gr.Textbox(placeholder="Ask Mistral...", container=False),
        title="EDU CHAT BY PHANINDRA REDDY K",
        examples=[
            "Explain quantum physics simply",
            "Write a haiku about technology",
            "What's the meaning of life?"
        ],
        cache_examples=False,
    )
    
    with gr.Row():
        audio = gr.Audio(sources=["microphone"], type="numpy", label="Voice Input")
        transcribe_btn = gr.Button("Send Transcription")
    
    transcribe_btn.click(
        transcribe,
        inputs=audio,
        outputs=chat_interface.textbox
    )

if __name__ == "__main__":
    demo.launch(share=True,debug=True)
