import os
import re
import uuid
import requests
import streamlit as st

# -----------------------------
# MUST be first
# -----------------------------
st.set_page_config(
    page_title="AI Merchant Intelligence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# API (Swagger: POST /ask body {"question": "..."} )
# -----------------------------
API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8001")
ASK_URL = os.getenv("API_URL", f"{API_BASE}/ask")

# -----------------------------
# Minimal, professional styling (don’t over-CSS)
# -----------------------------
CSS = """
<style>
/* Layout: center like ChatGPT */
.block-container { max-width: 920px; padding-top: 1.2rem; }

/* Sidebar: clean */
section[data-testid="stSidebar"]{
  border-right: 1px solid rgba(255,255,255,.08);
}

/* Title spacing */
h1, h2, h3 { letter-spacing: -0.02em; }

/* Reduce “boxy” feel on sidebar widgets */
section[data-testid="stSidebar"] .stButton button{
  width: 100%;
}

/* Make chat input feel premium */
div[data-testid="stChatInput"] textarea{
  border-radius: 14px !important;
}

/* Hide the little empty header band on top (optional) */
header[data-testid="stHeader"]{ background: transparent; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# -----------------------------
# Helpers
# -----------------------------
def make_topic(text: str, max_len: int = 32) -> str:
    t = text.strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^\w\s\-\&\?\/\.]", "", t)
    if not t:
        return "New chat"
    # take a short “subject-like” snippet
    words = t.split(" ")
    short = " ".join(words[:7]).strip()
    short = short[:1].upper() + short[1:]
    if len(short) > max_len:
        short = short[: max_len - 1].rstrip() + "…"
    return short

def call_api(question: str, timeout_s: int = 60) -> dict:
    payload = {"question": question}
    r = requests.post(ASK_URL, json=payload, timeout=timeout_s)
    r.raise_for_status()
    return r.json()

def extract_answer(data: dict) -> str:
    return (
        data.get("answer")
        or data.get("response")
        or data.get("result")
        or data.get("message")
        or str(data)
    )

# -----------------------------
# State
# -----------------------------
if "chats" not in st.session_state:
    # {chat_id: {"topic": str, "messages": [{"role": "user|assistant", "content": str}]}}
    st.session_state.chats = {}

if "current_chat_id" not in st.session_state:
    cid = str(uuid.uuid4())
    st.session_state.current_chat_id = cid
    st.session_state.chats[cid] = {"topic": "New chat", "messages": []}

def current_chat():
    return st.session_state.chats[st.session_state.current_chat_id]

# -----------------------------
# Sidebar: only what matters
# -----------------------------
with st.sidebar:
    st.markdown("## Chats")

    if st.button("➕ New chat"):
        cid = str(uuid.uuid4())
        st.session_state.current_chat_id = cid
        st.session_state.chats[cid] = {"topic": "New chat", "messages": []}
        st.rerun()

    ids = list(st.session_state.chats.keys())

    def label(cid: str) -> str:
        topic = st.session_state.chats[cid].get("topic", "New chat")
        return topic if topic else "New chat"

    if ids:
        chosen = st.radio(
            "History",
            options=ids,
            format_func=label,
            index=ids.index(st.session_state.current_chat_id),
            label_visibility="collapsed",
        )
        st.session_state.current_chat_id = chosen

    # optional: small rename (clean)
    st.markdown("---")
    st.caption("Rename current chat")
    chat = current_chat()
    new_name = st.text_input(
        "topic",
        value=chat.get("topic", "New chat"),
        label_visibility="collapsed",
        placeholder="Topic name…",
    )
    if new_name and new_name != chat.get("topic"):
        chat["topic"] = new_name.strip()

    # optional: clear (still useful, but not “settings”)
    if st.button("🧹 Clear messages"):
        chat["messages"] = []
        st.rerun()

# -----------------------------
# Main header (simple, professional)
# -----------------------------
st.markdown("# AI Merchant Intelligence")
st.caption("Hybrid Structured + RAG Financial AI Service")

# -----------------------------
# Render chat using native Streamlit chat components (looks more pro)
# -----------------------------
chat = current_chat()
msgs = chat["messages"]

if not msgs:
    st.info("Ask about payments, risk, chargebacks, fraud signals…", icon="💬")

for m in msgs:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# -----------------------------
# Input
# -----------------------------
prompt = st.chat_input("Message AI Merchant Intelligence…")

if prompt:
    # user msg
    msgs.append({"role": "user", "content": prompt})

    # auto topic on first user message
    if chat.get("topic", "New chat") in ("New chat", "", None):
        chat["topic"] = make_topic(prompt)

    # assistant msg
    try:
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                data = call_api(prompt, timeout_s=60)
                answer = extract_answer(data)
                st.markdown(answer)
        msgs.append({"role": "assistant", "content": answer})

    except requests.exceptions.RequestException as e:
        err = f"❌ API call failed: {e}"
        with st.chat_message("assistant"):
            st.error(err)
        msgs.append({"role": "assistant", "content": err})

    st.rerun()