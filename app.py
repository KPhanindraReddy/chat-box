import time
import gradio as gr
import pandas as pd
import openvino_genai
from huggingface_hub import snapshot_download
from threading import Lock, Event
import os
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import cpuinfo
import openvino as ov
import librosa
from googleapiclient.discovery import build
import gc
from PyPDF2 import PdfReader
from docx import Document
import textwrap
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
from typing import Generator


GOOGLE_API_KEY = "AIzaSyAo-1iW5MEZbc53DlEldtnUnDaYuTHUDH4"
GOOGLE_CSE_ID = "3027bedf3c88a4efb"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_NUM_IMAGES = 1
MAX_HISTORY_TURNS = 3
MAX_TOKENS_LIMIT = 4096

class UnifiedAISystem:
    def __init__(self):
        self.pipe_lock = Lock()
        self.current_df = None
        self.mistral_pipe = None
        self.internvl_pipe = None
        self.whisper_pipe = None
        self.current_document_text = None
        self.generation_executor = ThreadPoolExecutor(max_workers=3)
        self.initialize_models()

    def initialize_models(self):
        """Initialize all required models"""
        
        if not os.path.exists("mistral-ov"):
            snapshot_download(repo_id="OpenVINO/mistral-7b-instruct-v0.1-int8-ov", local_dir="mistral-ov")
        if not os.path.exists("internvl-ov"):
            snapshot_download(repo_id="OpenVINO/InternVL2-1B-int8-ov", local_dir="internvl-ov")
        if not os.path.exists("whisper-ov-model"):
            snapshot_download(repo_id="OpenVINO/whisper-tiny-fp16-ov", local_dir="whisper-ov-model")

        cpu_features = cpuinfo.get_cpu_info()['flags']
        config_options = {}
        if 'avx512' in cpu_features:
            config_options["ENFORCE_BF16"] = "YES"
        elif 'avx2' in cpu_features:
            config_options["INFERENCE_PRECISION_HINT"] = "f32"

        # Initialize Mistral model
        self.mistral_pipe = openvino_genai.LLMPipeline(
            "mistral-ov",
            device="CPU",
            config={"PERFORMANCE_HINT": "THROUGHPUT", **config_options}
        )

    
        self.whisper_pipe = openvino_genai.WhisperPipeline("whisper-ov-model", device="CPU")

    def load_data(self, file_path):
        """Load student data from file"""
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext == '.csv':
                self.current_df = pd.read_csv(file_path)
            elif file_ext in ['.xlsx', '.xls']:
                self.current_df = pd.read_excel(file_path)
            else:
                return False, "❌ Unsupported file format. Please upload a .csv or .xlsx file."
            return True, f"✅ Loaded {len(self.current_df)} records from {os.path.basename(file_path)}"
        except Exception as e:
            return False, f"❌ Error loading file: {str(e)}"

    def extract_text_from_document(self, file_path):
        """Extract text from PDF or DOCX documents"""
        text = ""
        try:
            file_ext = os.path.splitext(file_path)[1].lower()

            if file_ext == '.pdf':
                with open(file_path, 'rb') as file:
                    pdf_reader = PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"

            elif file_ext == '.docx':
                doc = Document(file_path)
                for para in doc.paragraphs:
                    text += para.text + "\n"

            else:
                return False, "❌ Unsupported document format. Please upload PDF or DOCX."

            # Clean and format text
            text = text.replace('\x0c', '')  
            text = textwrap.dedent(text)      
            self.current_document_text = text
            return True, f"✅ Extracted text from {os.path.basename(file_path)}"

        except Exception as e:
            return False, f"❌ Error processing document: {str(e)}"

    def generate_text_stream(self, prompt: str, max_tokens: int) -> Generator[str, None, None]:
        """Unified text generation with queued token streaming"""
        start_time = time.time()
        response_queue = Queue()
        completion_event = Event()
        error = [None]  

        optimized_config = openvino_genai.GenerationConfig(
            max_new_tokens=max_tokens,
            temperature=0.3,
            top_p=0.9,
            streaming=True,
            streaming_interval=5  
        )

        def callback(tokens): 
            response_queue.put("".join(tokens))
            return openvino_genai.StreamingStatus.RUNNING

        def generate():
            try:
                with self.pipe_lock:
                    self.mistral_pipe.generate(prompt, optimized_config, callback)
            except Exception as e:
                error[0] = str(e)
            finally:
                completion_event.set()

       
        self.generation_executor.submit(generate)

        accumulated = []
        token_count = 0
        last_gc = time.time()

        while not completion_event.is_set() or not response_queue.empty():
            if error[0]:
                yield f"❌ Error: {error[0]}"
                print(f"Stream generation time: {time.time() - start_time:.2f} seconds")
                return

            try:
                token_batch = response_queue.get(timeout=0.1)
                accumulated.append(token_batch)
                token_count += len(token_batch)
                yield "".join(accumulated)

              
                if time.time() - last_gc > 2.0:
                    gc.collect()
                    last_gc = time.time()
            except Empty:
                continue

        print(f"Generated {token_count} tokens in {time.time() - start_time:.2f} seconds "
              f"({token_count/(time.time() - start_time):.2f} tokens/sec)")
        yield "".join(accumulated)

    def analyze_student_data(self, query, max_tokens=500):
        """Analyze student data using AI with streaming"""
        if not query or not query.strip():
            yield "⚠️ Please enter a valid question"
            return

        if self.current_df is None:
            yield "⚠️ Please upload and load a student data file first"
            return

        data_summary = self._prepare_data_summary(self.current_df)
        prompt = f"""You are an expert education analyst. Analyze the following student performance data:
        {data_summary}
        Question: {query}
        Please include:
        1. Direct answer to the question
        2. Relevant statistics
        3. Key insights
        4. Actionable recommendations
        Format the output with clear headings"""

        
        yield from self.generate_text_stream(prompt, max_tokens)

    def _prepare_data_summary(self, df):
        """Summarize the uploaded data"""
        summary = f"Student performance data with {len(df)} rows and {len(df.columns)} columns.\n"
        summary += "Columns: " + ", ".join(df.columns) + "\n"
        summary += "First 3 rows:\n" + df.head(3).to_string(index=False)
        return summary

    def analyze_image(self, image, url, prompt):
        """Analyze image with InternVL model (synchronous, no streaming)"""
        try:
            if image is not None:
                image_source = image
            elif url and url.startswith(("http://", "https://")):
                response = requests.get(url)
                image_source = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                return "⚠️ Please upload an image or enter a valid URL"

            
            image_data = np.array(image_source.getdata()).reshape(
                1, image_source.size[1], image_source.size[0], 3
            ).astype(np.byte)
            image_tensor = ov.Tensor(image_data)

            
            if self.internvl_pipe is None:
                self.internvl_pipe = openvino_genai.VLMPipeline("internvl-ov", device="CPU")

            with self.pipe_lock:
                self.internvl_pipe.start_chat()
                output = self.internvl_pipe.generate(prompt, image=image_tensor, max_new_tokens=100)
                self.internvl_pipe.finish_chat()

           
            return output

        except Exception as e:
            return f"❌ Error: {str(e)}"

    def process_audio(self, data, sr):
        """Process audio data for speech recognition"""
        try:
            
            if data.ndim > 1:
                data = np.mean(data, axis=1)  
            else:
                data = data

           
            data = data.astype(np.float32)
            max_val = np.max(np.abs(data)) + 1e-7
            data /= max_val

            # Simple noise reduction
            data = np.clip(data, -0.5, 0.5)

            # Trim silence
            energy = np.abs(data)
            threshold = np.percentile(energy, 25)  
            mask = energy > threshold
            indices = np.where(mask)[0]

            if len(indices) > 0:
                start = max(0, indices[0] - 1000)
                end = min(len(data), indices[-1] + 1000)
                data = data[start:end]

          
            if sr != 16000:
                
                new_length = int(len(data) * 16000 / sr)
              
                data = np.interp(
                    np.linspace(0, len(data)-1, new_length),
                    np.arange(len(data)),
                    data
                )
                sr = 16000

            return data
        except Exception as e:
            print(f"Audio processing error: {e}")
            return np.array([], dtype=np.float32)

    def transcribe(self, audio):
        """Transcribe audio using Whisper model with improved error handling"""
        if audio is None:
            return ""
        sr, data = audio

      
        if len(data)/sr < 0.5:
            return ""

        try:
            processed = self.process_audio(data, sr)

            
            if len(processed) < 8000:  
                return ""

          
            result = self.whisper_pipe.generate(processed)
            return result
        except Exception as e:
            print(f"Transcription error: {e}")
            return "❌ Transcription failed - please try again"

    def generate_lesson_plan(self, topic, duration, additional_instructions="", max_tokens=1200):
        """Generate a lesson plan based on document content"""
        if not topic:
            yield "⚠️ Please enter a lesson topic"
            return

        if not self.current_document_text:
            yield "⚠️ Please upload and process a document first"
            return

        prompt = f"""As an expert educator, create a focused lesson plan using the provided content.
        **Core Requirements:**
        1. TOPIC: {topic}
        2. TOTAL DURATION: {duration} periods
        3. ADDITIONAL INSTRUCTIONS: {additional_instructions or 'None'}
        **Content Summary:**
        {self.current_document_text[:2500]}... [truncated]
        **Output Structure:**
        1. PERIOD ALLOCATION (Break topic into {duration} logical segments):
          - Period 1: [Subtopic 1]
          - Period 2: [Subtopic 2]
             ...
        2. LEARNING OBJECTIVES (Max 3 bullet points)
        3. TEACHING ACTIVITIES (One engaging method per period)
        4. RESOURCES (Key materials from document)
        5. ASSESSMENT (Simple checks for understanding)
        6. PAGE REFERENCES (Specific source pages)
**Key Rules:**
- Strictly divide content into exactly {duration} periods
- Prioritize document content over creativity
- Keep objectives measurable
- Use only document resources
- Make page references specific"""

     
        yield from self.generate_text_stream(prompt, max_tokens)

    def fetch_images(self, query: str, num: int = DEFAULT_NUM_IMAGES) -> list:
        """Fetch unique images by requesting different result pages"""
        if num <= 0:
            return []

        try:
            service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
            image_links = []
            seen_urls = set()  

            
            for start_index in range(1, num * 2, 2):
                if len(image_links) >= num:
                    break

                res = service.cse().list(
                    q=query,
                    cx=GOOGLE_CSE_ID,
                    searchType="image",
                    num=1,
                    start=start_index
                ).execute()

                if "items" in res and res["items"]:
                    item = res["items"][0]
                    # Skip duplicates
                    if item["link"] not in seen_urls:
                        image_links.append(item["link"])
                        seen_urls.add(item["link"])

            return image_links[:num]
        except Exception as e:
            print(f"Error in image fetching: {e}")
            return []


ai_system = UnifiedAISystem()


css = """
    :root {
  --bg: #0D0D0D;
  --surface: #1F1F1F;
  --primary: #BB86FC;
  --secondary: #03DAC6;
  --accent: #CF6679;
  --success: #4CAF50;
  --warning: #FFB300;
  --text: #FFFFFF;
  --subtext: #B0B0B0;
  --divider: #333333;
}
body, .gradio-container { background: var(--bg); color: var(--text); }
.user-msg,
.bot-msg,
.upload-box,
#question-input,
.mode-checkbox,
.system-info,
.lesson-plan { background: var(--surface); border-radius: 8px; color: var(--text); }
.user-msg,
.bot-msg { padding: 12px 16px; margin: 8px 0; line-height:1.5; border-left:4px solid var(--primary); box-shadow:0 2px 6px rgba(0,0,0,0.5); }
.bot-msg { border-color: var(--secondary); }
.upload-box { padding:16px; margin-bottom:16px; border:1px solid var(--divider); }
#question-input,
.mode-checkbox { padding:12px; border:1px solid var(--divider); }
.slider-container { margin:20px 0; padding:15px; border-radius:10px; background:var(--secondary); }
.system-info { padding:15px; margin:15px 0; border-left:4px solid var(--primary); }
.chat-image { max-height:100px; margin:4px; border-radius:8px; box-shadow:0 2px 6px rgba(0,0,0,0.5); cursor:pointer; transition:transform .2s; }
.chat-image:hover { transform:scale(1.05); box-shadow:0 4px 10px rgba(0,0,0,0.7); }
.modal { position:fixed; inset:0; background:rgba(0,0,0,0.9); display:none; cursor:zoom-out; }
.modal-content { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); max-width:90%; max-height:90%; padding:10px; border-radius:12px; background:var(--surface); }
.modal-img { max-width:100%; max-height:100%; border-radius:8px; }
.typing-indicator { display:inline-block; position:relative; width:40px; height:20px; }
.typing-dot { width:6px; height:6px; border-radius:50%; background:var(--text); position:absolute; animation:typing 1.4s infinite ease-in-out; }
.typing-dot:nth-child(1){left:0;}
.typing-dot:nth-child(2){left:12px;animation-delay:.2s}
.typing-dot:nth-child(3){left:24px;animation-delay:.4s}
@keyframes typing{0%,60%,100%{transform:translateY(0)}30%{transform:translateY(-5px)}}
.lesson-title { font-size:1.2em; font-weight:bold; color:var(--primary); margin-bottom:8px; }
.page-ref { display:inline-block; padding:3px 8px; margin:3px; border-radius:4px; background:var(--primary); color:var(--text); font-size:.9em; }
/* Scrollbar */
.chatbot::-webkit-scrollbar{width:8px}
.chatbot::-webkit-scrollbar-track{background:var(--surface);border-radius:4px}
.chatbot::-webkit-scrollbar-thumb{background:var(--primary);border-radius:4px}
.chatbot::-webkit-scrollbar-thumb:hover{background:var(--secondary)}
"""


with gr.Blocks(css=css, title="Unified EDU Assistant") as demo:
    gr.Markdown("# 🤖 Unified EDU Assistant by Phanindra Reddy K")

   
    gr.HTML("""
    <div class="system-info">
        <strong>Multi-Modal AI Assistant</strong>
        <ul>
            <li>Text & Voice Chat with Mistral-7B</li>
            <li>Image Understanding with InternVL</li>
            <li>Student Data Analysis</li>
            <li>Visual Search with Google Images</li>
            <li>Lesson Planning from Documents</li>
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

    chat_state = gr.State([])
    with gr.Column(scale=2, elem_classes="chat-container"):
        chatbot = gr.Chatbot(label="Conversation", height=500, bubble_full_width=False,
                            avatar_images=("user.png", "bot.png"), show_label=False)

  
    with gr.Row():
        chat_mode = gr.Checkbox(label="💬 General Chat", value=True, elem_classes="mode-checkbox")
        student_mode = gr.Checkbox(label="🎓 Student Analytics", value=False, elem_classes="mode-checkbox")
        image_mode = gr.Checkbox(label="🖼️ Image Analysis", value=False, elem_classes="mode-checkbox")
        lesson_mode = gr.Checkbox(label="📝 Lesson Planning", value=False, elem_classes="mode-checkbox")

    
    with gr.Column() as chat_inputs:
        include_images = gr.Checkbox(label="Include Visuals", value=True)
        user_input = gr.Textbox(
            placeholder="Type your question here...",
            label="Your Question",
            container=False,
            elem_id="question-input"
        )
        with gr.Row():
            max_tokens = gr.Slider(
                minimum=10,
                maximum=1000,
                value=100,
                step=10,
                label="Response Length (Tokens)"
            )
            num_images = gr.Slider(
                minimum=0,
                maximum=5,
                value=1,
                step=1,
                label="Number of Images",
                visible=True
            )


    with gr.Column(visible=False) as student_inputs:
        file_upload = gr.File(label="CSV/Excel File", file_types=[".csv", ".xlsx"], type="filepath")
        student_question = gr.Textbox(
            placeholder="Ask questions about student data...",
            label="Your Question",
            elem_id="question-input"
        )
        student_status = gr.Markdown("No file loaded")

  
    with gr.Column(visible=False) as image_inputs:
        image_upload = gr.Image(type="pil", label="Upload Image")
        image_url = gr.Textbox(
            label="OR Enter Image URL",
            placeholder="https://example.com/image.jpg",
            elem_id="question-input"
        )
        image_question = gr.Textbox(
            placeholder="Ask questions about the image...",
            label="Your Question",
            elem_id="question-input"
        )


    with gr.Column(visible=False) as lesson_inputs:
        gr.Markdown("### 📚 Lesson Planning")
        doc_upload = gr.File(
            label="Upload Curriculum Document (PDF/DOCX)",
            file_types=[".pdf", ".docx"],
            type="filepath"
        )
        doc_status = gr.Markdown("No document uploaded")

        with gr.Row():
            topic_input = gr.Textbox(
                label="Lesson Topic",
                placeholder="Enter the main topic for the lesson plan"
            )
            duration_input = gr.Number(
                label="Total Periods",
                value=5,
                minimum=1,
                maximum=20,
                step=1
            )

        additional_instructions = gr.Textbox(
            label="Additional Requirements (optional)",
            placeholder="Specific teaching methods, resources, or special considerations..."
        )

        generate_btn = gr.Button("Generate Lesson Plan", variant="primary")


    with gr.Row():
        submit_btn = gr.Button("Send", variant="primary")
        mic_btn = gr.Button("Transcribe Voice", variant="secondary")
        mic = gr.Audio(sources=["microphone"], type="numpy", label="Voice Input")

  
    def toggle_modes(chat, student, image, lesson):
        return [
            gr.update(visible=chat),
            gr.update(visible=student),
            gr.update(visible=image),
            gr.update(visible=lesson)
        ]

    def load_student_file(file_path):
        success, message = ai_system.load_data(file_path)
        return message

    def process_document(file_path):
        if not file_path:
            return "⚠️ Please select a document first"
        success, message = ai_system.extract_text_from_document(file_path)
        return message

    def render_history(history):
        """Render chat history with images and proper formatting"""
        rendered = []
        for user_msg, bot_msg, image_links in history:
            user_html = f"<div class='user-msg'>{user_msg}</div>"

           
            bot_text = str(bot_msg)

            if "Lesson Plan:" in bot_text:
                bot_html = f"<div class='lesson-plan'>{bot_text}</div>"
            else:
                bot_html = f"<div class='bot-msg'>{bot_text}</div>"

            # Add images if available
            if image_links:
                images_html = "".join(
                    f"<img src='{url}' class='chat-image' onclick='showImage(\"{url}\")' />"
                    for url in image_links
                )
                bot_html += f"<br><br><b>📸 Related Visuals:</b><br><div style='display: flex; flex-wrap: wrap;'>{images_html}</div>"

            rendered.append((user_html, bot_html))
        return rendered

    def respond(message, history, chat, student, image, lesson,
               tokens, student_q, image_q, image_upload, image_url,
               include_visuals, num_imgs, topic, duration, additional):
        """
        1. Use actual_message (depending on mode) instead of raw `message`.
        2. Convert any non‐string Bot response (like VLMDecodedResults) to str().
        3. Disable the input box during streaming, then re-enable it at the end.
        """
        updated_history = list(history)

       
        if student:
            actual_message = student_q
        elif image:
            actual_message = image_q
        elif lesson:
            actual_message = f"Generate lesson plan for: {topic} ({duration} periods)"
            if additional:
                actual_message += f"\nAdditional: {additional}"
        else:
            actual_message = message

       
        typing_html = "<div class='typing-indicator'><div class='typing-dot'></div><div class='typing-dot'></div><div class='typing-dot'></div></div>"
        updated_history.append((actual_message, typing_html, []))

        yield render_history(updated_history), gr.update(value="", interactive=False), updated_history

        full_response = ""
        images = []

        try:
            if chat:
                
                for chunk in ai_system.generate_text_stream(actual_message, tokens):
                    full_response = chunk
                    updated_history[-1] = (actual_message, full_response, [])
                    yield render_history(updated_history), gr.update(value="", interactive=False), updated_history

                if include_visuals:
                    images = ai_system.fetch_images(actual_message, num_imgs)

            elif student:
               
                if ai_system.current_df is None:
                    full_response = "⚠️ Please upload a student data file first"
                else:
                    for chunk in ai_system.analyze_student_data(student_q, tokens):
                        full_response = chunk
                        updated_history[-1] = (actual_message, full_response, [])
                        yield render_history(updated_history), gr.update(value="", interactive=False), updated_history

            elif image:
               
                if (not image_upload) and (not image_url):
                    full_response = "⚠️ Please upload an image or enter a URL"
                else:
                    
                    result_obj = ai_system.analyze_image(image_upload, image_url, image_q)
                    full_response = str(result_obj)

            elif lesson:
               
                if not topic:
                    full_response = "⚠️ Please enter a lesson topic"
                else:
                    duration = int(duration) if duration else 5
                    for chunk in ai_system.generate_lesson_plan(topic, duration, additional, tokens):
                        full_response = chunk
                        updated_history[-1] = (actual_message, full_response, [])
                        yield render_history(updated_history), gr.update(value="", interactive=False), updated_history

           
            updated_history[-1] = (actual_message, full_response, images)
            if len(updated_history) > MAX_HISTORY_TURNS:
                updated_history = updated_history[-MAX_HISTORY_TURNS:]

        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            updated_history[-1] = (actual_message, error_msg, [])

        
        yield render_history(updated_history), gr.update(value="", interactive=True), updated_history

    # Voice transcription
    def transcribe_audio(audio):
        return ai_system.transcribe(audio)

   
    chat_mode.change(fn=toggle_modes, inputs=[chat_mode, student_mode, image_mode, lesson_mode],
                   outputs=[chat_inputs, student_inputs, image_inputs, lesson_inputs])
    student_mode.change(fn=toggle_modes, inputs=[chat_mode, student_mode, image_mode, lesson_mode],
                      outputs=[chat_inputs, student_inputs, image_inputs, lesson_inputs])
    image_mode.change(fn=toggle_modes, inputs=[chat_mode, student_mode, image_mode, lesson_mode],
                    outputs=[chat_inputs, student_inputs, image_inputs, lesson_inputs])
    lesson_mode.change(fn=toggle_modes, inputs=[chat_mode, student_mode, image_mode, lesson_mode],
                     outputs=[chat_inputs, student_inputs, image_inputs, lesson_inputs])

    # File upload handler
    file_upload.change(fn=load_student_file, inputs=file_upload, outputs=student_status)

    # Document upload handler
    doc_upload.change(fn=process_document, inputs=doc_upload, outputs=doc_status)

    mic_btn.click(fn=transcribe_audio, inputs=mic, outputs=user_input)

    # Submit handler
    submit_btn.click(
        fn=respond,
        inputs=[
            user_input, chat_state, chat_mode, student_mode, image_mode, lesson_mode,
            max_tokens, student_question, image_question, image_upload, image_url,
            include_images, num_images,
            topic_input, duration_input, additional_instructions
        ],
        outputs=[chatbot, user_input, chat_state]
    )

   
    generate_btn.click(
        fn=respond,
        inputs=[
            gr.Textbox(value="Generate lesson plan", visible=False),  
            chat_state,
            chat_mode, student_mode, image_mode, lesson_mode,
            max_tokens,
            gr.Textbox(visible=False),  
            gr.Textbox(visible=False), 
            gr.Image(visible=False),    
            gr.Textbox(visible=False), 
            gr.Checkbox(visible=False), 
            gr.Slider(visible=False),   
            topic_input,                
            duration_input,             
            additional_instructions     
        ],
        outputs=[chatbot, user_input, chat_state]
    )

if __name__ == "__main__":
    demo.launch(share=True, debug=True)

