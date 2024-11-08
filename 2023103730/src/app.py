import streamlit as stream
from llm_chains import load_normal_chain
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

def load_chain(chat_history):
    return load_normal_chain(chat_history)

def clear_input_field():
    stream.session_state.user_question = stream.session_state.user_input
    stream.session_state.user_input = ""

def set_send_input():
    stream.session_state.send_input = True
    clear_input_field()

def main():
    stream.set_page_config(page_title = "chatbot for middle24projects", page_icon = "icon.png")
    chat_container = stream.container()
    
    if "send_input" not in stream.session_state:
        stream.session_state.send_input = False
        stream.session_state.user_question = ""
    
    chat_history = StreamlitChatMessageHistory(key = "history")
    llm_chain = load_chain(chat_history)
    
    user_input = stream.text_input("您有什么想问的吗？", key = "user_input", on_change = set_send_input)
    send_button = stream.button("Ask", key = "send_button")
    
    if send_button or stream.session_state.send_input:
        if stream.session_state.user_question != "":
        
            with chat_container:
                stream.chat_message("user").write(stream.session_state.user_question)
                llm_response = llm_chain.run(stream.session_state.user_question)
                stream.chat_message("ai").write(llm_response)
                stream.session_state.user_question = ""
                stream.session_state.send_input = False 
    
    if chat_history.messages != []:
        stream.write("Chat History:")
        
        for message in chat_history.messages:
            stream.chat_message(message.type).write(message.content)
        

if __name__ == "__main__":
    main()