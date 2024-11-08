import streamlit as st
import json
from streamlit_chat import message
from movie_chatbot import MovieChatbot

# st.set_page_config(page_title="Movie Chatbot", layout="wide")

api_file_path = "/data4/myt/MovieChat/api_prompt/api.json"
prompt_path = "/data4/myt/MovieChat/api_prompt/prompt.json"
video_path = "/data5/yzh/DATASETS/Movie101/video/6965768652251628068.mp4"

with open(api_file_path, "r") as f:
    openai_api_key = json.load(f)["open_ai_key"]
    
st.title("Movie Chatbot")


st.video(video_path, format="video/mp4", start_time=0)


with st.form("chat_input", clear_on_submit=True):
    a, b = st.columns([4, 1])
    user_input = a.text_input(
        label="Your message:",
        placeholder="What would you like to say?",
        label_visibility="collapsed",
    )
    b.form_submit_button("Send", use_container_width=True)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
    
if openai_api_key:
    chat_bot = MovieChatbot(openai_api_key, prompt_path, video_path)

if user_input:
    st.session_state.messages.insert(0, {"role": "user", "content": user_input})
    response = chat_bot.UserRequest(st.session_state.messages[::-1])
    st.session_state.messages.insert(0, {"role": "assistant", "content": response})
    
key_id = 0
if st.session_state["messages"]:
    for msg in st.session_state.messages:
        message(msg["content"], is_user=msg["role"] == "user", key=key_id)
        key_id += 1
    