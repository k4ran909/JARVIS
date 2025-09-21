import os
import json
import tempfile
import streamlit as st
from langchain_ollama import ChatOllama
import torch
import io 
import base64

# ----- Custom Modules -----
from src.FUNCTION.run_function import FunctionExecutor
from src.BRAIN.text_to_info import send_to_ai
from src.BRAIN.local_func_call import LocalFunctionCall
from src.CONVERSATION.text_to_speech import speak
from src.FUNCTION.Tools.random_respon import RandomChoice
from src.FUNCTION.Tools.greet_time import TimeOfDay
from DATA.msg import WELCOME_RESPONSES
from src.BRAIN.gem_func_call import GeminiFunctionCaller
from src.BRAIN.RAG import RAGPipeline
from src.BRAIN.chat_with_ai import PersonalChatAI
from src.CONVERSATION.text_speech import text_to_speech_local
from src.CONVERSATION.voice_text import voice_to_text
from src.BRAIN.code_gen import CodeRefactorAssistant
from src.VISION.eye import ImageProcessor  # <-- your image class

# # ----- Torch fix -----
if hasattr(torch.classes, '__path__'):
    torch.classes.__path__ = []

# ----- Initialize Components -----
local_caller = LocalFunctionCall()
func_executor = FunctionExecutor()
time_greeter = TimeOfDay()
code_assistant = CodeRefactorAssistant()
chat_ai = PersonalChatAI()
rag = RAGPipeline()

# ----- Streamlit config -----
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
UPLOAD_DIR = "."
os.makedirs(UPLOAD_DIR, exist_ok=True)

AI_MODEL = "granite3.1-dense:2b"

# ----- Session Initialization -----
def initialize_session():
    defaults = {
        "chat_mode": "normal",
        "chat_histories": {
            "normal": [],
            "chat_with_ai": [],
            "chat_with_rag": [],
            "data_analysis": [],
            "image_processing": []
        },
        "rag_subject": "",
        "voice_output": False,
        "uploaded_file_path": None,
        "audio_input_key_counter": 0,
        "image_path": None,
        "image_obj": None,
        "image_action": None
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def set_greeting():
    if "greeted_once" not in st.session_state:
        st.session_state.greeted_once = False
    if not st.session_state.greeted_once:
        greeting_message = f"{time_greeter.time_of_day()}. {RandomChoice.random_choice(WELCOME_RESPONSES)}"
        st.session_state.greeting_message = greeting_message
        st.session_state.greeted_once = True


initialize_session()
set_greeting()
if "greeting_message" in st.session_state:
    speak(st.session_state.greeting_message)
    del st.session_state["greeting_message"]

# ----- Utility Functions -----
@st.cache_resource(show_spinner=False)
def load_rag_chain(subject):
    try:
        return rag.setup_chain(subject.lower().strip().replace(" ", "_"))
    except:
        return None

@st.cache_data(show_spinner=False)
def personal_chat_ai(query, max_token=2000):
    try:
        messages = chat_ai.message_management(query)
        llm = ChatOllama(model=AI_MODEL, temperature=0.3, max_token=max_token)
        response_content = "".join(chunk.content for chunk in llm.stream(messages))
        chat_ai.store_important_chat(query, response_content)
        return response_content
    except Exception as e:
        return f"An error occurred: {e}"

def data_analysis(user_prompt: str, file_path: str):
    try:
        return code_assistant.gem_text_to_code(user_prompt, file_path)
    except:
        return code_assistant.local_text_to_code(user_prompt, file_path)

def chat_with_rag_session(subject, query):
    key = subject.lower().strip().replace(" ", "_")
    if f"qa_chain_{key}" not in st.session_state:
        st.session_state[f"qa_chain_{key}"] = load_rag_chain(key)
    qa_chain = st.session_state.get(f"qa_chain_{key}")
    return rag.ask(qa_chain, query) if qa_chain else f"Error: Unable to load RAG chain for '{subject}'."

def process_command(command):
    try:
        gem_caller = GeminiFunctionCaller()
        response_list_dic = gem_caller.generate_function_calls(command)
        if not response_list_dic:
            raise ValueError("Empty Gemini output.")
    except:
        response_list_dic = local_caller.create_function_call(command)
        if not response_list_dic:
            raise ValueError("Empty Local model output.")
        
    results = []
    for dic in response_list_dic:
        func_name = dic.get("name")
        args = dic.get("arguments", {})
        speak(f"Executing function: {func_name}")
        try:
            res = func_executor.execute(dic)
            results.append(res)
        except Exception as e:
            results.append({"status": "failed", "function_name": func_name, "args": args, "output": str(e)})
    send_to_ai(f"Respond to user's command '{command}' concisely.")
    return results

def add_message(role, content):
    history = st.session_state.chat_histories[st.session_state.chat_mode]
    if not any(msg["content"] == content and msg["role"] == role for msg in history):
        history.append({"role": role, "content": content})

# ----- Sidebar: Mode Selection -----
st.sidebar.markdown("### 🔁 Select Chat Mode")
mode_display_map = {
    "💬 Normal": "normal",
    "🧍 Personal Chat": "chat_with_ai",
    "📚 RAG Chat": "chat_with_rag",
    "📊 Data Analysis": "data_analysis",
    "🖼️ Image Processing": "image_processing"
}
selected_display = st.sidebar.selectbox("🧠 Select Chat Mode", list(mode_display_map.keys()))
selected_mode = mode_display_map[selected_display]
st.session_state.chat_mode = selected_mode

# Voice toggle
st.session_state.voice_output = st.sidebar.toggle("🎙️ Voice Reply", value=st.session_state.voice_output)

# RAG topic select
if st.session_state.chat_mode == "chat_with_rag":
    st.session_state.rag_subject = st.sidebar.selectbox("📘 Select RAG Topic", [
        "Disaster", "Finance", "Healthcare", "Artificial Intelligence", "Climate Change",
        "Cybersecurity", "Education", "Space Technology", "Politics", "History", "Biology"
    ])

# Data Analysis upload
if st.session_state.chat_mode == "data_analysis":
    st.sidebar.markdown("### 📤 Upload CSV File")
    uploaded_file = st.sidebar.file_uploader("Choose CSV", type=["csv"])
    if uploaded_file:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.uploaded_file_path = file_path
        st.sidebar.success(f"✅ File saved to: `{file_path}`")

# Image Processing setup
if st.session_state.chat_mode == "image_processing":
    st.sidebar.markdown("### 🖼️ Image Input")
    image_source = st.sidebar.radio("Image Source", ["Upload", "URL", "Camera"])
    processor = ImageProcessor()
    if image_source == "Upload":
        uploaded_img = st.sidebar.file_uploader("Upload Image", type=["png","jpg","jpeg"])
        if uploaded_img:
            path = os.path.join(UPLOAD_DIR, uploaded_img.name)
            with open(path, "wb") as f: f.write(uploaded_img.getbuffer())
            processor.image_path = path
            st.session_state.image_path = path
            st.image(path, caption="Uploaded Image")
    elif image_source == "URL":
        url = st.sidebar.text_input("Image URL")
        if url:
            img = processor.get_image_from_url(url)
            st.session_state.image_obj = img
            processor.image_obj = img
            st.image(img, caption="Fetched Image")
    elif image_source == "Camera":
        if st.sidebar.button("📸 Capture Image"):
            captured = processor.capture_image_and_save()
            if captured:
                st.session_state.image_path = captured
                processor.image_path = captured
                st.image(captured, caption="Captured Image")

    st.session_state.image_action = st.sidebar.selectbox(
        "Select Action", ["Basic Detection", "Object Detection", "Segmentation", "Resize"]
    )

# ----- Current Mode Display -----
st.markdown(f"<div style='text-align:center; font-size:30px;'>🧠 <b>Current Mode:</b> {st.session_state.chat_mode.upper()}</div>", unsafe_allow_html=True)

# ----- Show Chat History -----
for msg in st.session_state.chat_histories[st.session_state.chat_mode]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)
        if "image" in msg and msg["image"]:
            st.image(
                io.BytesIO(base64.b64decode(msg["image"])),
                use_column_width=True
            )

# ----- Chat Input -----
if user_input := st.chat_input("Ask me anything... "):
    st.chat_message("user").markdown(user_input)
    add_message("user", user_input)

    # --- Mode handling ---
    if st.session_state.chat_mode == "chat_with_ai":
        response = personal_chat_ai(user_input)
    elif st.session_state.chat_mode == "chat_with_rag":
        subject = st.session_state.rag_subject
        response = chat_with_rag_session(subject, user_input) if subject else "🚨 Enter a subject."
    elif st.session_state.chat_mode == "data_analysis":
        if not st.session_state.uploaded_file_path:
            response = "🚨 Please upload a CSV first."
        else:
            response = data_analysis(user_input, st.session_state.uploaded_file_path)
    elif st.session_state.chat_mode == "image_processing":
        if not processor.image_path and not processor.image_obj:
            response = "🚨 Please provide an image first."
        else:
            action = st.session_state.image_action
            if action == "Basic Detection":
                response = processor.detect_image(query=user_input) or "No result."
            elif action == "Object Detection":
                boxes = processor.detect_objects(prompt=user_input)
                response = f"Detected boxes: {boxes}" if boxes else "No objects detected."
            elif action == "Segmentation":
            # Extract and return composite image for display
                segmented_image = processor.extract_segmentation_masks(
                    output_dir="segmentation_results",
                    prompt=user_input
                )
                if segmented_image:
                    # Add to chat history with an 'image' key
                    st.session_state.chat_histories[st.session_state.chat_mode].append({
                        "role": "assistant",
                        "content": "Segmentation complete.",
                        "image": segmented_image
                    })
                    response = "Segmentation complete. Image displayed above."
                else:
                    st.session_state.chat_histories[st.session_state.chat_mode].append({
                        "role": "assistant",
                        "content": "Segmentation failed or no objects detected.",
                        "image": None
                    })
                    response = "Segmentation failed or no objects detected."

                #response = "Segmentation masks saved in `segmentation_results/`."
            elif action == "Resize":
                success = processor.resize_image()
                response = f"Image resized to {processor.require_width}x{processor.require_height}" if success else "Resize failed."
    else:
        response = process_command(user_input)

    # --- Display response ---
    if isinstance(response, list) and isinstance(response[0], dict):
        for entry in response:
            with st.chat_message("assistant"):
                st.markdown(f"""
                **🔧 Function Executed:** `{entry.get('function_name', 'N/A')}`  
                **📌 Status:** ✅ `{entry.get('status', 'unknown')}`  
                **📂 Arguments:** `{entry.get('args', {})}`  
                **📜 Output:** *{entry.get('output', 'No output.')}*
                """)
        add_message("assistant", json.dumps(response, indent=2))
    else:
        st.chat_message("assistant").markdown(response)
        add_message("assistant", response)

    # --- Voice output ---
    if st.session_state.voice_output:
        full_resp = "\n".join([r.get("output","") for r in response]) if isinstance(response,list) else str(response)
        audio_io = text_to_speech_local(full_resp.replace("*",""))
        st.markdown(f"""
            <audio autoplay="true">
                <source src="data:audio/mp3;base64,{audio_io}" type="audio/mp3">
            </audio>
        """, unsafe_allow_html=True)





#------------------- Voice Input & Audio Handling -------------------
audio_key = f"audio_input_key_{st.session_state.audio_input_key_counter}"
audio_value = st.sidebar.audio_input("🎤 Speak", key=audio_key)

if audio_value:
    with tempfile.TemporaryFile(suffix=".wav") as temp_audio:
        temp_audio.write(audio_value.getvalue())
        temp_audio.seek(0)
        transcribed_text = voice_to_text(temp_audio)

    if transcribed_text:
        st.chat_message("user").markdown(transcribed_text)
        add_message("user", transcribed_text)

        if st.session_state.chat_mode == "chat_with_ai":
            response = personal_chat_ai(transcribed_text)
        elif st.session_state.chat_mode == "chat_with_rag":
            subject = st.session_state.rag_subject
            response = chat_with_rag_session(subject, transcribed_text) if subject else "🚨 Please enter a subject."
        elif st.session_state.chat_mode == "data_analysis":
            try:
                if not st.session_state.uploaded_file_path:
                    response = "🚨 Please upload a CSV file first."
                else:
                    file_path = st.session_state.uploaded_file_path
                    response = data_analysis(transcribed_text,file_path)
            except Exception as e:
                response = f"Error reading CSV: {e}"
        else:
            response = process_command(transcribed_text)
        
        if isinstance(response, list) and isinstance(response[0], dict):
            for entry in response:
                with st.chat_message("assistant"):
                    st.markdown(f"""
                    **🔧 Function Executed:** `{entry.get('function_name', 'N/A')}`  
                    **📌 Status:** ✅ `{entry.get('status', 'unknown')}`  
                    **📂 Arguments:** `{entry.get('args', {})}`  
                    **📜 Output:** *{entry.get('output', 'No output.')}*
                    """)
            add_message("assistant", json.dumps(response, indent=2))
        else:
            st.chat_message("assistant").markdown(response)
            add_message("assistant", response)

        # st.chat_message("assistant").markdown(str(response))
        # add_message("assistant", str(response))

        if st.session_state.voice_output:
            full_response = "\n".join([sub_response.get("output", "") for sub_response in response]) if isinstance(response, list) else str(response)
            audio_io = text_to_speech_local(full_response.replace("*", ""))
            st.markdown(f"""
                <audio autoplay="true">
                    <source src="data:audio/mp3;base64,{audio_io}" type="audio/mp3">
                </audio>
            """, unsafe_allow_html=True)

        del st.session_state[audio_key]
        st.session_state.audio_input_key_counter += 1


# ----- Download History -----
st.sidebar.download_button(
    label="📅 Download Chat History",
    data=json.dumps(st.session_state.chat_histories[st.session_state.chat_mode], indent=2),
    file_name="chat_history.json",
    mime="application/json"
)
