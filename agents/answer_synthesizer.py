from typing import List, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from config import Config

SYSTEM_PROMPT = """
You are SU_BOT — the AI assistant for Sarvajanik College of Engineering and Technology (SCET), Surat.
Use the given context to answer questions accurately. Cite your sources clearly.
If unsure, say so politely. Be concise and factual.
"""

def _mk_context_block(items: List[Dict]) -> str:
    if not items:
        return "No context."
    lines = []
    for i, it in enumerate(items, 1):
        lines.append(f"[{i}] ({it.get('kind','')}) {it.get('title','')}\n{it.get('content','')}")
    return "\n\n".join(lines)

def _mk_citations(items: List[Dict]) -> str:
    cites = []
    for it in items:
        if it["kind"] == "web" and it["url"]:
            cites.append(f"- [{it['title']}]({it['url']})")
        elif it["kind"] == "local":
            cites.append(f"- {it['source']}")
    return "\n".join(cites)

def synthesize_answer(query: str, model_name: str, local_ctx: List[Dict], web_ctx: List[Dict]) -> str:
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=Config.GEMINI_API_KEY,
        temperature=0.3,
        max_output_tokens=1024
    )

    merged = (local_ctx or []) + (web_ctx or [])
    context_block = _mk_context_block(merged)
    citations = _mk_citations(merged)

    prompt = f"""{SYSTEM_PROMPT}

User Question:
{query}

Context:
{context_block}

Sources:
{citations if citations else "- None"}
"""
    resp = llm.invoke(prompt)
    return resp.content if hasattr(resp, "content") else str(resp)
