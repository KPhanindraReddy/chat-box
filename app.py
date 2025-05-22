from huggingface_hub import snapshot_download
import openvino_genai
import gradio as gr
from threading import Thread, Lock,Event
from queue import Queue, Empty

# Download model
model_dir = snapshot_download(
    repo_id="OpenVINO/Mistral-7B-Instruct-v0.2-int4-ov",
    local_dir="mistral-ov",
    token=False
)

# Initialize pipeline and config once
pipe = openvino_genai.LLMPipeline(model_dir, device="CPU")
config = openvino_genai.GenerationConfig(
    max_new_tokens=100,
    num_beams=1,
    do_sample=False,
    temperature=0.4
)
pipe_lock = Lock()

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
                pipe.start_chat()
                pipe.generate(message, config, callback)
                pipe.finish_chat()
        except Exception as e:
            error_message[0] = str(e)
        finally:
            completion_event.set()

    Thread(target=generate).start()

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

    # Final yield to ensure complete response
    yield "".join(accumulated)

# Create Gradio interface
demo = gr.ChatInterface(
    stream_generator,
    textbox=gr.Textbox(placeholder="Ask Mistral...", container=False),
    title="Mistral-7B Chat",
    examples=[
        "Explain quantum physics simply",
        "Write a haiku about technology",
        "What's the meaning of life?"
    ],
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch(share=True, debug=True) 