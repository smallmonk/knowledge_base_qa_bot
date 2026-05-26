import os

from langchain.schema import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from . import indexer


SYSTEM_PROMPT = """You are a helpful knowledge base Q&A assistant.

Your task is to answer the user's questions based ONLY on the provided CONTEXT.
Follow these strict guidelines:
1. Grounding: Rely ONLY on the information explicitly stated in the CONTEXT. Do not use any outside knowledge, do not assume, and do not guess.
2. Citation: You MUST cite the source of your information. For every fact or statement you make, append the source identifier in the format `[filename#heading]` (e.g., `[refund_policy.md#refund-timeline]`). Do not cite if the information is not in that source.
3. Fallback: If the CONTEXT does not contain the answer to the user's question, you must answer exactly: "I cannot confirm from the knowledge base." Do not attempt to answer or make up any information.
4. No guessing or speculation: If the information is incomplete, state that you cannot confirm it.
"""

_llm = None


def get_llm():
    global _llm
    if _llm is None:
        model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
        api_key = os.getenv("GEMINI_API_KEY","")
        
        _llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            max_retries=1,
        )
    return _llm


def build_prompt(query: str, ranked_sections: list) -> str:
    context_parts = []
    for section, score in ranked_sections:
        heading_path_str = " > ".join(section.heading_path)
        context_parts.append(
            f"[Source: {section.id}]\nHeading Path: {heading_path_str}\nContent:\n{section.content}"
        )
    context_str = "\n\n---\n\n".join(context_parts)
    return f"CONTEXT:\n{context_str}\n\nQUESTION:\n{query}"


def query(question: str) -> dict:
    if not indexer.sections:
        return {
            "answer": "The knowledge base has not been indexed yet. Call POST /index first.",
            "sources": [],
        }

    ranked_sections = indexer.search(question, k=3)
    if not ranked_sections:
        return {
            "answer": "I cannot confirm from the knowledge base.",
            "sources": [],
        }

    response = get_llm().invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=build_prompt(question, ranked_sections)),
    ])

    sources = [
        {
            "source": section.id,
            "heading": " > ".join(section.heading_path),
            "score": round(score, 3),
            "content": section.content[:240],
        }
        for section, score in ranked_sections
    ]

    return {
        "answer": response.content,
        "sources": sources,
    }
