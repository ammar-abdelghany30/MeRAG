"""app.py - Streamlit front-end for MeRAG."""
import streamlit as st

from src.qa import TOP_K, answer
from src.vectordb_creation import INDEX_FILE, search

st.set_page_config(page_title="MeRAG", page_icon="🔎", layout="wide")

EXAMPLES = [
    "Where did he intern and what did he work on?",
    "What are his main technical skills?",
    "Tell me about the SkillGap-AI project",
    "What does he study and where?",
]

# Stop early with a clear message if the index hasn't been built yet
if not INDEX_FILE.exists():
    st.error("No index found. Run `python -m src.vectordb_creation` first.")
    st.stop()


@st.cache_resource(show_spinner="Loading embedding model and index...")
def warm_up():
    search("warm up", k=1)  # loads the model and index once, then they stay in memory
    return True


warm_up()

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- Sidebar ----------
with st.sidebar:
    st.title("🔎 MeRAG")
    st.caption(
        "Ask about my background. Answers are grounded in my CV, "
        "LinkedIn and GitHub, and every answer shows its sources."
    )
    k = st.slider("Chuncks to retrieve (k)", 1, 10, TOP_K)
    show_chunks = st.toggle("Show retrieved passages", value=True)

    st.divider()
    st.subheader("Try an example")
    for q in EXAMPLES:
        if st.button(q, use_container_width=True):
            st.session_state.pending = q

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------- Helpers ----------
def render_assistant(msg):
    st.markdown(msg["content"])
    if msg.get("sources"):
        st.caption("Sources: " + "  ·  ".join(msg["sources"]))
    if show_chunks and msg.get("chunks"):
        with st.expander(f"Retrieved passages ({len(msg['chunks'])})"):
            for i, c in enumerate(msg["chunks"], start=1):
                st.markdown(f"**[{i}] {c['label']}**  ·  score {c['score']:.2f}")
                st.text(c["text"])


# ---------- Main page ----------
st.title("Ask MeRAG")
st.caption("A personal RAG system: retrieve the relevant passages, answer only from them, cite the source.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_assistant(msg)

prompt = st.chat_input("Ask a question about my background...")
prompt = prompt or st.session_state.pop("pending", None)  # example-button click

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching my documents and thinking..."):
                out = answer(prompt, k=k)
            msg = {
                "role": "assistant",
                "content": out["answer"],
                "sources": out["sources"],
                "chunks": out.get("chunks", []),
            }
            render_assistant(msg)
            st.session_state.messages.append(msg)
        except Exception as e:
            st.error(f"Something went wrong: {e}")