import streamlit as st
from config import Config
from tools.web_search import web_search
from retriever.local_index import build_or_load_local_retriever, retrieve_local
from retriever.router import pick_route
from agents.controller import make_plan
from agents.answer_synthesizer import synthesize_answer

st.set_page_config(page_title="SU_BOT Agentic RAG", page_icon="🤖", layout="wide")
st.title("🤖 SU_BOT — Agentic RAG (Hybrid: Local + Web)")

try:
    Config.validate()
except ValueError as e:
    st.error(str(e))
    st.stop()

st.sidebar.header("⚙️ Settings")
model = st.sidebar.selectbox("Gemini Model", ["gemini-1.5-pro", "gemini-2.0-flash"], index=0)
web_k = st.sidebar.slider("Web Results", 3, 10, 5)

# Local retriever
@st.cache_resource(show_spinner=False)
def _load_local():
    return build_or_load_local_retriever("data/scet")

local_retriever = _load_local()
has_local = local_retriever is not None

if st.sidebar.button("♻️ Rebuild Local Index"):
    build_or_load_local_retriever("data/scet")
    st.sidebar.success("Local SCET index rebuilt successfully!")

if "chat" not in st.session_state:
    st.session_state.chat = []

for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

q = st.chat_input("Ask about SCET departments, faculty, placements, etc.")
if q:
    st.session_state.chat.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            route_hint = pick_route(q, has_local=has_local)
            plan = make_plan(q, has_local, route_hint.use_web, route_hint.use_local)

            local_ctx, web_ctx = [], []
            if plan.route in ("local", "hybrid") and has_local:
                local_ctx = retrieve_local(q, local_retriever)
            if plan.route in ("web", "hybrid"):
                web_ctx = web_search(q, max_results=web_k)

            ans = synthesize_answer(q, model, local_ctx, web_ctx)
            st.markdown(ans)
            st.session_state.chat.append({"role": "assistant", "content": ans})

st.markdown("<hr><center>Built by Karan Mehta | Gemini + Tavily + OpenAI Embeddings</center>", unsafe_allow_html=True)
