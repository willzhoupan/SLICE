
from ftllm import llm
import sys
import os
import argparse
from util import make_normal_parser
from util import make_normal_llm_model

def parse_args():
    parser = make_normal_parser("fastllm webui")
    parser.add_argument("--port", type = int, default = 8080, help = "API server port")
    parser.add_argument("--title", type = str, default = "fastllm webui", help = "页面标题")
    return parser.parse_args()

args = parse_args()

import streamlit as st
from streamlit_chat import message
st.set_page_config(
    page_title = args.title,
    page_icon = ":robot:"
)

@st.cache_resource
def get_model():
    args = parse_args()
    model = make_normal_llm_model(args)
    model.set_verbose(True)
    return model

if "messages" not in st.session_state:
    st.session_state.messages = []

system_prompt = st.sidebar.text_input("system_prompt", "")
max_new_tokens = st.sidebar.slider("max_new_tokens", 0, 8192, 512, step = 1)
top_p = st.sidebar.slider("top_p", 0.0, 1.0, 0.8, step = 0.01)
top_k = st.sidebar.slider("top_k", 1, 50, 1, step = 1)
temperature = st.sidebar.slider("temperature", 0.0, 10.0, 1.0, step = 0.1)
repeat_penalty = st.sidebar.slider("repeat_penalty", 1.0, 10.0, 1.0, step = 0.05)

buttonClean = st.sidebar.button("清理会话历史", key="clean")
if buttonClean:
    st.session_state.messages = []
    st.rerun()

for i, (prompt, response) in enumerate(st.session_state.messages):
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        st.markdown(response)

if prompt := st.chat_input("请开始对话"):
    model = get_model()
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        messages = []
        if system_prompt != "":
            messages.append({"role": "system", "content": system_prompt})
        for his in st.session_state.messages:
            messages.append({"role": "user", "content": his[0]})
            messages.append({"role": "assistant", "content": his[1]})
        messages.append({"role": "user", "content": prompt})

        for chunk in model.stream_response(messages,
                                           max_length = max_new_tokens,
                                           top_k = top_k,
                                           top_p = top_p,
                                           temperature = temperature,
                                           repeat_penalty = repeat_penalty,
                                           one_by_one = True):
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append((prompt, full_response))
