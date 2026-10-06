import logging
from llama_index.core.schema import NodeWithScore
from llama_index.llms.ollama import Ollama
from src.indexer import load_index
from src.prompts import CLIENT_PROMPT, MANAGER_PROMPT
from src.settings import settings

logger = logging.getLogger(__name__)

logger.info("Загрузка индекса из %s", settings.chroma_dir)
_index = load_index()
_retriever = _index.as_retriever(similarity_top_k=settings.retriever_top_k)

_llm = Ollama(
    model=settings.ollama_model,
    base_url=settings.ollama_base_url,
    temperature=settings.temperature,
    request_timeout=120.0,   # локальная модель на CPU может отвечать 20-40 сек
    context_window=8192,
)

logger.info(
    "RAG готов: model=%s, top_k=%d",
    settings.ollama_model,
    settings.retriever_top_k,
)

def retrieve_context(query: str) -> tuple[str, list[str], list[str]]:
    """
    Возвращает:
      context  — склеенный текст top-k нод (для обоих промптов),
      upsells  — список подсказок по допродажам из metadata,
      sources  — список id найденных записей (для UI и отладки).
    """
    nodes: list[NodeWithScore] = _retriever.retrieve(query)
    context_parts: list[str] = []
    upsells: list[str] = []
    sources: list[str] = []

    for n in nodes:
        context_parts.append(n.text)
        sources.append(n.metadata.get("id", "unknown"))
        upsell = n.metadata.get("upsell")
        if upsell and upsell not in upsells:
            upsells.append(upsell)

    context = "\n\n".join(context_parts)
    logger.debug("Получено %d нод: %s", len(nodes), sources)
    return context, upsells, sources

def generate_client_reply(query: str, context: str, history_text: str) -> str:
    prompt = CLIENT_PROMPT.format(
        context=context,
        query=query,
        history=history_text,
    )
    return str(_llm.complete(prompt)).strip()

def generate_manager_hint(
    query: str,
    context: str,
    upsells: list[str],
    history_text: str,
) -> str:
    upsell_text = "\n".join(f"- {u}" for u in upsells) if upsells else "нет данных"
    prompt = MANAGER_PROMPT.format(
        context=context,
        upsells=upsell_text,
        query=query,
        history=history_text,
    )
    return str(_llm.complete(prompt)).strip()

def process_inquiry(query: str, history: list[dict] | None = None) -> dict:
    context, upsells, sources = retrieve_context(query)
    history_text = format_history(history)
    
    client_reply = generate_client_reply(query, context, history_text)
    manager_hint = generate_manager_hint(query, context, upsells, history_text)
    
    return {
        "client_reply": client_reply,
        "manager_hint": manager_hint,
        "sources": sources,
    }

def format_history(history: list[dict] | None, max_messages: int = 6) -> str:
    """
    Превращает список сообщений Gradio в читаемый текст для промпта.
    Ограничивает длину, чтобы не раздувать контекст LLM.
    """
    if not history:
        return "(диалог только начался)"
    
    recent = history[-max_messages:]
    lines = []
    for msg in recent:
        speaker = "Клиент" if msg.get("role") == "user" else "Менеджер"
        content = _extract_text(msg.get("content", "")).strip()
        if content:
            lines.append(f"{speaker}: {content}")
    
    return "\n".join(lines) if lines else "(диалог только начался)"

def _extract_text(content) -> str:
    """
    Достаёт текстовую часть из content.
    Поддерживает оба формата Gradio: строку и список частей.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            # item может быть dict {"type": "text", "text": "..."} или строкой
            if isinstance(item, dict):
                text = item.get("text", "")
                if text:
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return " ".join(parts)
    return ""