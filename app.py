import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
# ** MODIFIED: Updated import for HuggingFaceEmbeddings **
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# --- CONFIGURATION ---
PERSIST_DIRECTORY = "db_chroma"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# --- END CONFIGURATION ---

def load_environment():
    """Loads environment variables from .env file."""
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("Google API Key not found. Please create a .env file with GOOGLE_API_KEY='your_key'")
        st.stop()

def initialize_models_and_db():
    """Initializes the embedding model, language model, and vector database."""
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
        
        # ** MODIFIED: Updated the model name from "gemini-pro" to "gemini-1.0-pro" **
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.7, convert_system_message_to_human=True)
        return db, llm
    except Exception as e:
        st.error(f"Error initializing models or database: {e}")
        st.info("Please make sure you have run 'python ingest.py' successfully to create the database.")
        st.stop()

def create_qa_chain(db, llm):
    """Creates the Question-Answering chain."""
    retriever = db.as_retriever(search_kwargs={"k": 3})

    template = """
    You are a helpful and friendly assistant for Sarvajanik College of Engineering and Technology (SCET).
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
    load_environment()
    
    st.set_page_config(page_title="SCET Chatbot", page_icon="🎓")
    st.title("🎓 SCET University Chatbot")
    st.caption("Ask me anything about Sarvajanik College of Engineering and Technology!")

    db, llm = initialize_models_and_db()
    qa_chain = create_qa_chain(db, llm)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What would you like to know?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = qa_chain({"query": prompt})
                st.markdown(response["result"])

                with st.expander("Sources"):
                    for source_doc in response["source_documents"]:
                        st.write(f"- {source_doc.metadata['source']}")
        
        st.session_state.messages.append({"role": "assistant", "content": response["result"]})

if __name__ == "__main__":
    main()