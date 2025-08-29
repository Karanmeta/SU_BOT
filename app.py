import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import torch
import os

# --- CONFIGURATION ---
PERSIST_DIRECTORY = "db_chroma"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.1"
# --- END CONFIGURATION ---

def initialize_models_and_db():
    """Initializes the embedding model, language model, and vector database."""
    try:
        # Check if database exists
        if not os.path.exists(PERSIST_DIRECTORY):
            st.error("❌ Chroma database not found!")
            st.info("➡️ Run `python ingest_from_local.py` first to create the database from your local files.")
            st.stop()
        
        # Check database files
        db_files = os.listdir(PERSIST_DIRECTORY)
        if not any(f.endswith('.sqlite3') for f in db_files):
            st.error("❌ Database appears to be empty or corrupted!")
            st.info("➡️ Run `python ingest_from_local.py` to recreate the database.")
            st.stop()
        
        # Smart device detection
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.sidebar.write(f"**Device:** `{device}`")
        
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": device}
        )
        
        db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
        
        # Verify database has content
        collection = db.get()
        if len(collection['ids']) == 0:
            st.error("❌ Database is empty!")
            st.info("➡️ Run `python ingest_from_local.py` to add your documents.")
            st.stop()
        
        # Ollama local model
        try:
            llm = Ollama(model=OLLAMA_MODEL, temperature=0.7)
            return db, llm
        except Exception as e:
            st.error(f"❌ Error initializing Ollama: {e}")
            st.info("➡️ Make sure Ollama is installed and running: `ollama serve`")
            st.info("➡️ Download the model: `ollama pull llama3.1`")
            st.stop()
            
    except Exception as e:
        st.error(f"❌ Error initializing database: {e}")
        st.info("➡️ Run `python ingest_from_local.py` first to create the vector database.")
        st.stop()

def create_qa_chain(db, llm):
    """Creates the Question-Answering chain."""
    retriever = db.as_retriever(search_kwargs={"k": 3})

    template = """
    You are a helpful and friendly assistant for Sarvajanik University.
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't have enough information, don't try to make up an answer.
    Provide a detailed and well-formatted answer.
    
    Context: {context}
    
    Question: {question}
    
    Helpful Answer:
    """
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    return RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        return_source_documents=True
    )

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Sarvajanik University Chatbot", page_icon="🎓")
    st.title("🎓 Sarvajanik University Chatbot")
    st.caption("Ask me anything about Sarvajanik University!")

    # Sidebar info
    with st.sidebar:
        st.header("ℹ️ Info")
        st.write(f"**Embedding model:** `{EMBEDDING_MODEL}`")
        st.write(f"**Ollama model:** `{OLLAMA_MODEL}`")
        st.markdown("---")
        st.info("✅ Make sure Ollama is running in background")

    db, llm = initialize_models_and_db()
    qa_chain = create_qa_chain(db, llm)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("What would you like to know?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = qa_chain({"query": prompt})
                    answer = response.get("result", "⚠️ No answer generated.")
                except Exception as e:
                    answer = f"❌ Error generating response: {e}"
                    response = {"source_documents": []}

                st.markdown(answer)

                # Show sources safely
                sources = response.get("source_documents", [])
                if sources:
                    with st.expander("📂 Sources"):
                        for source_doc in sources:
                            st.write(f"- {source_doc.metadata.get('source', 'Unknown source')}")

        st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()