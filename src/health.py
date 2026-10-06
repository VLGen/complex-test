import logging
import httpx
from src.settings import settings

logger = logging.getLogger(__name__)

def is_ollama_alive(timeout: float = 3.0) -> bool:
    """Проверяет, отвечает ли Ollama на /api/tags."""
    try:
        r = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=timeout)
        return r.status_code == 200
    except httpx.RequestError:
        return False

def list_ollama_models(timeout: float = 3.0) -> list[str]:
    """Возвращает список имён моделей, доступных в Ollama."""
    try:
        r = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=timeout)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except (httpx.RequestError, httpx.HTTPStatusError):
        return []

def is_model_available(model_name: str) -> bool:
    """
    Проверяет наличие модели. Учитывает, что Ollama хранит имена
    в формате 'qwen2.5:7b'
    """
    models = list_ollama_models()
    if not models:
        return False
    # Точное совпадение или совпадение по префиксу до ':'
    base = model_name.split(":")[0]
    return any(m == model_name or m.split(":")[0] == base for m in models)

def ping_llm(timeout: float = 60.0) -> tuple[bool, str]:
    """
    Отправляет крошечный промпт в Ollama.
    """
    try:
        r = httpx.post(
            f"{settings.ollama_base_url}/api/generate",
            json={
                "model": settings.ollama_model,
                "prompt": "Ответь одним словом: ок",
                "stream": False,
            },
            timeout=timeout,
        )
        r.raise_for_status()
        answer = r.json().get("response", "").strip()
        return True, answer or "(пустой ответ)"
    except httpx.HTTPStatusError as e:
        return False, f"HTTP {e.response.status_code}: {e.response.text[:200]}"
    except httpx.RequestError as e:
        return False, f"Сетевая ошибка: {e}"

def check_environment() -> str:
    """
    Возвращает Markdown-строку со статусом для отображения в UI.
    """
    if not is_ollama_alive():
        logger.warning("Ollama недоступна по адресу %s", settings.ollama_base_url)
        return (
            f"❌ **Ollama недоступна** по адресу `{settings.ollama_base_url}`.\n\n"
            "Запустите сервис командой `ollama serve` и повторите проверку."
        )

    if not is_model_available(settings.ollama_model):
        available = list_ollama_models()
        available_str = ", ".join(f"`{m}`" for m in available) or "нет моделей"
        logger.warning("Модель %s не найдена. Доступно: %s", settings.ollama_model, available)
        return (
            f"❌ **Модель `{settings.ollama_model}` не найдена** в Ollama.\n\n"
            f"Выполните: `ollama pull {settings.ollama_model}`\n\n"
            f"Доступные модели: {available_str}"
        )

    return (
        f"✅ **Ollama работает.** Модель `{settings.ollama_model}` доступна "
        f"по адресу `{settings.ollama_base_url}`."
    )