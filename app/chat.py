"""
Chat service — Ollama LLM + RAG retrieval.

Provides streaming chat responses augmented with knowledge retrieved
from the ChromaDB vector store.
"""

from typing import AsyncGenerator, Optional

import ollama as ollama_lib

from app.rag import retrieve

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "llama3.1:8b"
OLLAMA_URL = "http://127.0.0.1:11434"

SYSTEM_PROMPT = """\
You are a knowledgeable plant pathology assistant specializing in plant disease \
identification, treatment, and prevention. Your purpose is to help gardeners, \
farmers, and plant enthusiasts care for their plants.

Guidelines:
- Provide clear, actionable advice based on the provided context documents.
- Distinguish between chemical, organic, and cultural management options.
- When discussing chemical treatments, include safety reminders (follow label \
  directions, observe pre-harvest intervals).
- If you are unsure about something, say so rather than guessing.
- Stay on topic: only answer questions related to plants, gardening, \
  agriculture, plant diseases, and plant care.
- If asked about topics unrelated to plants and agriculture, politely redirect \
  the conversation.
- Cite the source documents when referencing specific information.
- Be concise but thorough.
"""

DIAGNOSIS_CONTEXT_TEMPLATE = """\
The user has just received a plant diagnosis from our AI classifier:
- Plant: {plant}
- Condition: {condition}
- Confidence: {confidence}%
- Class: {class_name}
{healthy_note}
Please use this context when answering their questions. They are likely asking \
about this specific diagnosis.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_system_prompt(
    diagnosis_context: Optional[dict] = None,
    rag_context: Optional[list[dict]] = None,
) -> str:
    """Assemble the full system prompt with optional diagnosis + RAG context."""
    parts = [SYSTEM_PROMPT]

    if diagnosis_context:
        healthy_note = (
            "The plant appears healthy."
            if diagnosis_context.get("healthy")
            else "The plant has been diagnosed with a disease."
        )
        parts.append(
            DIAGNOSIS_CONTEXT_TEMPLATE.format(
                plant=diagnosis_context.get("plant", "Unknown"),
                condition=diagnosis_context.get("condition", "Unknown"),
                confidence=diagnosis_context.get("confidence", "N/A"),
                class_name=diagnosis_context.get("class_name", "Unknown"),
                healthy_note=healthy_note,
            )
        )

    if rag_context:
        context_text = "\n\n---\n\n".join(
            f"[Source: {doc['title']}]\n{doc['text']}" for doc in rag_context
        )
        parts.append(
            "Relevant knowledge base documents for reference:\n\n"
            + context_text
            + "\n\nUse the above documents to inform your response."
        )

    return "\n\n".join(parts)


def _build_messages(
    user_message: str,
    system_prompt: str,
    chat_history: Optional[list[dict]] = None,
) -> list[dict]:
    """Build the messages list for the Ollama chat API.

    Assembles [system, ...history, user] in the format expected by
    Ollama's /api/chat endpoint.  History is capped at the last 20
    messages to stay within typical context-window limits.

    Args:
        user_message: The current user query.
        system_prompt: Full system prompt (including RAG context).
        chat_history: Prior conversation turns, each with 'role' and 'content'.

    Returns:
        List of message dicts ready for ``ollama.chat(messages=...)``.
    """
    messages = [{"role": "system", "content": system_prompt}]

    if chat_history:
        for msg in chat_history[-20:]:  # Limit history to last 20 messages
            messages.append({
                "role": msg["role"],
                "content": msg["content"],
            })

    messages.append({"role": "user", "content": user_message})
    return messages


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def generate_response(
    message: str,
    diagnosis_context: Optional[dict] = None,
    chat_history: Optional[list[dict]] = None,
    model: str = DEFAULT_MODEL,
    ollama_url: str = OLLAMA_URL,
) -> AsyncGenerator[str, None]:
    """
    Generate a streaming chat response augmented with RAG context.

    Pipeline: query → retrieve top-5 chunks → build system prompt
    with RAG context + diagnosis info → stream tokens from Ollama.

    Args:
        message: The user's question.
        diagnosis_context: Optional dict with plant/condition/confidence
            from the image classifier, injected so the LLM can reference
            the current diagnosis.
        chat_history: Prior conversation turns for multi-turn context.
        model: Ollama model identifier (default ``llama3.1:8b``).
        ollama_url: Base URL of the Ollama server.

    Yields:
        str: Individual text tokens as they arrive from the LLM.
    """
    # 1. Retrieve relevant knowledge
    rag_results = retrieve(query=message, n_results=5, ollama_url=ollama_url)

    # 2. Build system prompt with context
    system_prompt = _build_system_prompt(
        diagnosis_context=diagnosis_context,
        rag_context=rag_results,
    )

    # 3. Build message list
    messages = _build_messages(message, system_prompt, chat_history)

    # 4. Stream from Ollama
    client = ollama_lib.AsyncClient(host=ollama_url)
    stream = await client.chat(
        model=model,
        messages=messages,
        stream=True,
    )

    async for chunk in stream:
        text = chunk.get("message", {}).get("content", "")
        if text:
            yield text


async def check_ollama_status(
    ollama_url: str = OLLAMA_URL,
) -> dict:
    """Check if Ollama is running and return available models."""
    try:
        client = ollama_lib.AsyncClient(host=ollama_url)
        models_response = await client.list()
        model_names = [m.model for m in models_response.models]
        return {
            "available": True,
            "models": model_names,
            "default_model": DEFAULT_MODEL,
        }
    except Exception as e:
        return {
            "available": False,
            "models": [],
            "default_model": DEFAULT_MODEL,
            "error": str(e),
        }


async def list_models(
    ollama_url: str = OLLAMA_URL,
) -> list[str]:
    """List available Ollama models."""
    try:
        client = ollama_lib.AsyncClient(host=ollama_url)
        models_response = await client.list()
        return [m.model for m in models_response.models]
    except Exception:
        return []
