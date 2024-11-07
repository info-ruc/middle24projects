import streamlit as st
import json
import mimetypes
from streamlit_chat import message
from movie_chatbot import MovieChatbot
from base64 import b64encode
from pathlib import Path
from streamlit_player import st_player

def _media_file_base64(file_path, mime='video/mp4', start_time=0):
    """
    Helper func that returns base64 of the media file

    :param file_path: path of the file
    :param mime: mime type
    :param start_time: start time

    :return: base64 of the media file
    """
    if file_path == '':
        data = ''
        return [{"type": mime, "src": f"data:{mime};base64,{data}#t={start_time}"}]
    with open(file_path, "rb") as media_file:
        data = b64encode(media_file.read()).decode()
        try:
            mime = mimetypes.guess_type(file_path)[0]
        except Exception as e:
            print(f'Unrecognized video type!')
    return [{"type": mime, "src": f"data:{mime};base64,{data}#t={start_time}"}]

st.set_page_config(page_title="Movie Chatbot", layout="wide")

api_file_path = "/data4/myt/MovieChat/api_prompt/api.json"
prompt_path = "/data4/myt/MovieChat/api_prompt/prompt.json"
video_path = "/data5/yzh/DATASETS/Movie101/video/6965768652251628068.mp4"
# video_path = "/data4/myt/MovieChat/mycomponent/1275_2101.mp4"   

with open(api_file_path, "r") as f:
    openai_api_key = json.load(f)["open_ai_key"]

if openai_api_key:
    chat_bot = MovieChatbot(openai_api_key, prompt_path, video_path)


col_l, col_r = st.columns([7, 3])
html_style = '''
<style>
div:has( >.element-container div.floating) {
    display: flex;
    flex-direction: column;
    position: fixed;
}

div.floating {
    height: 0%;
}
</style>
'''
st.markdown(html_style, unsafe_allow_html=True)


with col_l:
    st.markdown('<div class="floating"></div>', unsafe_allow_html=True)
    st.subheader("Movie Chatbot")
    
    options = {
        "playback_rate": 1,
        'config': {
            'file': {
                'attributes': {
                    'crossOrigin': 'true'
                }
            }}
    }
    video_data = _media_file_base64(video_path)
    # event = st_player(video_data, **options, height=500, key="player")
    st.video(video_path)


with col_r:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if user_input := st.chat_input("Chat about the movie?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with col_r:
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            response = chat_bot.UserRequest(st.session_state.messages)
            message_placeholder.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
    
