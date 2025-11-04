from langchain.retrievers import TavilySearchAPIRetriever

def get_hybrid_retriever(k: int = 5):
    """
    Returns a Tavily retriever for live, web-based search.
    """
    return TavilySearchAPIRetriever(k=k)
