from tavily import TavilyClient
from config import Config

def web_search(query: str, max_results: int = 5):
    client = TavilyClient(api_key=Config.TAVILY_API_KEY)
    results = client.search(query=query, max_results=max_results, search_depth="advanced")

    formatted = []
    for item in results.get("results", []):
        formatted.append({
            "title": item.get("title", "Untitled"),
            "url": item.get("url", ""),
            "content": (item.get("content", "") or "")[:1200],
            "source": item.get("url", ""),
            "kind": "web"
        })
    return formatted
