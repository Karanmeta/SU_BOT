import streamlit as st
from config import Config
from retriever.hybrid_retriever import get_hybrid_retriever
from memory.chat_memory import get_memory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI

# ------------------- App Setup -------------------
st.set_page_config(page_title="SU_BOT 4.0", page_icon="🤖", layout="wide")

st.markdown("""
<h1 style='text-align:center; color:#4B9CD3;'>🤖 SU_BOT 4.0 — Gemini-Powered Smart Assistant</h1>
<p style='text-align:center;'>Memory-enabled | Web-aware | Powered by Google Gemini 1.5 Pro</p>
""", unsafe_allow_html=True)

# ------------------- Initialize Session -------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

@st.cache_resource(show_spinner=False)
def load_bot():
    retriever = get_hybrid_retriever(k=5)
    memory = get_memory()

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        google_api_key=Config.GEMINI_API_KEY,
        temperature=0.3
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=False
    )
    return qa_chain

qa = load_bot()

# ------------------- Chat Interface -------------------
st.markdown("### 💬 Chat with SU_BOT")

# User input
user_query = st.chat_input("Ask me anything...")

# Display chat history
for i, (query, answer) in enumerate(st.session_state.chat_history):
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        st.markdown(answer)

# Handle new question
if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = qa.invoke({"question": user_query, "chat_history": st.session_state.chat_history})
                answer = result["answer"]
            except Exception as e:
                answer = f"⚠️ Error: {str(e)}"
            st.markdown(answer)
            st.session_state.chat_history.append((user_query, answer))

st.markdown("""
<hr>
<p style='text-align:center; font-size:13px; color:gray;'>
Built with ❤️ by Karan Mehta using Gemini 1.5 Pro, LangChain & Streamlit
</p>
""", unsafe_allow_html=True)
