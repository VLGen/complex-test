import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

import gradio as gr
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.health import check_environment, ping_llm
from src.rag import process_inquiry
from src.settings import settings

logger = logging.getLogger(__name__)
PLACEHOLDER_HINT = "*Подсказка появится после обработки обращения*"

def on_submit(message: str, chatbot_history: list[dict]):
    """
    Обрабатывает новое сообщение клиента.

    Args:
        message: текст от клиента
        chatbot_history: текущая история чата в формате messages

    Returns:
        обновлённая история, подсказка менеджеру, источники, очищенное поле ввода
    """
    # Пустой ввод — ничего не делаем
    if not message or not message.strip():
        return chatbot_history, gr.update(), gr.update(), gr.update()

    # Вызов ядра RAG
    result = process_inquiry(query=message, history=chatbot_history)

    # Добавляем в чат две реплики: клиент + рекомендованный ответ менеджера.
    # Роль "user" — клиент, "assistant" — менеджер (это Gradio отрисует как две стороны).
    chatbot_history = chatbot_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": result["client_reply"]},
    ]

    # Источники — в Markdown-список
    sources = result.get("sources", [])
    sources_md = "\n".join(f"- `{s}`" for s in sources) if sources else "_нет источников_"

    return chatbot_history, result["manager_hint"], sources_md, ""

def on_clear():
    """Полный сброс: чат, подсказка, источники, поле ввода."""
    return [], PLACEHOLDER_HINT, "", ""

def handle_ping():
    """Обёртка над ping_llm() для кнопки проверки генерации."""
    ok, msg = ping_llm()
    icon = "✅" if ok else "❌"
    return f"{icon} **Ping LLM:** {msg}"

with gr.Blocks(title="O-complex Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🧠 Ассистент менеджера O-complex")
    
    # Статус-бар (health check)
    status = gr.Markdown(check_environment())
    
    with gr.Row():
        # ЛЕВАЯ КОЛОНКА — диалог
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                height=500,
                label="Диалог с клиентом",
            )
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Введите сообщение клиента...",
                    show_label=False,
                    scale=4,
                )
                submit_btn = gr.Button("▶", variant="primary", scale=0)
            with gr.Row():
                clear_btn = gr.Button("🗑 Очистить диалог", size="sm")
                check_btn = gr.Button("🔄 Проверить Ollama", size="sm")
                ping_btn = gr.Button("🏓 Ping LLM", size="sm")
            
            gr.Examples(
                examples=[
                    "Сколько стоит доставка?",
                    "Что такое цеолит?",
                    "Можно ли принимать с витаминами?",
                ],
                inputs=msg_input,
                label="Примеры обращений",
            )
        
        # ПРАВАЯ КОЛОНКА — подсказки менеджеру
        with gr.Column(scale=1):
            manager_hint = gr.Markdown(
                value="*Подсказка появится после обработки обращения*",
                label="💡 Подсказка менеджеру",
            )
            with gr.Accordion("📚 Источники из базы знаний", open=False):
                sources_out = gr.Markdown()
    
    # Обработчики
    submit_btn.click(
        on_submit,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, manager_hint, sources_out, msg_input],
    )
    msg_input.submit(
        on_submit,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, manager_hint, sources_out, msg_input],
    )
    clear_btn.click(
        lambda: ([], "", "", ""),
        outputs=[chatbot, manager_hint, sources_out, msg_input],
    )
    ping_btn.click(
    fn=handle_ping,
    inputs=None,
    outputs=status,
    )
    # Повторная проверка Ollama
    check_btn.click(
        fn=check_environment,
        inputs=None,
        outputs=status,
    )

if __name__ == "__main__":
    logger.info("Запуск Gradio на %s:%d", settings.gradio_server_name, settings.gradio_server_port)
    demo.queue().launch(
        theme=gr.themes.Soft(),
        server_name=settings.gradio_server_name,
        server_port=settings.gradio_server_port,
        share=settings.gradio_share,
    )