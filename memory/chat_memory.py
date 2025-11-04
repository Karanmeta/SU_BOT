from langchain.memory import ConversationBufferMemory

def get_memory():
    """
    Returns a conversation memory buffer to preserve context across turns.
    """
    return ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        return_messages=True
    )
