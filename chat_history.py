import streamlit as st
import json
import os
from datetime import datetime
from pathlib import Path

from src.engine import get_chat_engine
from src.model_loader import initialise_llm, get_embedding_model

# -------------------------
# Setup
# -------------------------

CHAT_DIR = Path("chat_history")
CHAT_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title("📊 RAG Financial Chatbot")

# -------------------------
# Load Chat Engine
# -------------------------

@st.cache_resource
def load_engine():
    llm = initialise_llm()
    embed_model = get_embedding_model()
    return get_chat_engine(llm=llm, embed_model=embed_model)

chat_engine = load_engine()

# -------------------------
# Session State
# -------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_chat" not in st.session_state:
    st.session_state.current_chat = None

# -------------------------
# Sidebar - Saved Chats
# -------------------------

st.sidebar.title("💬 Saved Chats")

saved_chats = list(CHAT_DIR.glob("*.json"))

for chat_file in saved_chats:
    if st.sidebar.button(chat_file.stem):
        with open(chat_file, "r") as f:
            st.session_state.messages = json.load(f)
            st.session_state.current_chat = chat_file

# New Chat Button
if st.sidebar.button("➕ New Chat"):
    st.session_state.messages = []
    st.session_state.current_chat = None

# -------------------------
# Display Messages
# -------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -------------------------
# Chat Input
# -------------------------

if prompt := st.chat_input("Ask your question..."):

    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Bot response
    with st.chat_message("assistant"):
        response = chat_engine.chat(prompt)
        st.markdown(response.response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response.response}
    )

    # -------------------------
    # Save Chat
    # -------------------------

    if st.session_state.current_chat is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = CHAT_DIR / f"chat_{timestamp}.json"
        st.session_state.current_chat = filename
    else:
        filename = st.session_state.current_chat

    with open(filename, "w") as f:
        json.dump(st.session_state.messages, f, indent=2)