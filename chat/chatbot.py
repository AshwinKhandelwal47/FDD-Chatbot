import logging
from typing import Any, Callable

from src.chat.prompts import SYSTEM_PROMPT
from src.llm.factory import (
    ProviderUnavailableError,
    build_provider_order,
    format_provider_label,
    get_provider_chain,
)

logger = logging.getLogger(__name__)


def chat_with_fallback(
    messages: list[dict[str, str]],
    system: str | None = None,
    chat_provider: str | None = None,
) -> tuple[str, str]:
    """Chat using the provider fallback chain. Returns (response, provider_label)."""
    system_prompt = system if system is not None else SYSTEM_PROMPT
    provider_order = (
        build_provider_order(chat_provider) if chat_provider else None
    )
    providers = get_provider_chain("chat", provider_order=provider_order)
    last_error: ProviderUnavailableError | None = None

    for provider in providers:
        try:
            response = provider.chat(messages=messages, system=system_prompt)
            label = format_provider_label(provider)
            logger.info("Chat answered by %s", label)
            return response, label
        except ProviderUnavailableError as exc:
            logger.warning("Chat provider %s unavailable: %s", provider.name, exc)
            last_error = exc

    raise ProviderUnavailableError(
        f"All chat providers failed. Last error: {last_error}"
    )


def ask(
    query: str,
    history: list[dict[str, str]] | None = None,
    chat_provider: str | None = None,
) -> tuple[str, str]:
    """Ask a question with conversation history. Returns (answer, provider_label)."""
    messages = list(history or [])
    messages.append({"role": "user", "content": query})
    return chat_with_fallback(messages, chat_provider=chat_provider)


def ask_with_rag(
    query: str,
    history: list[dict[str, str]] | None = None,
    retriever: Callable[[str], list[dict[str, Any]]] | None = None,
    chat_provider: str | None = None,
) -> tuple[str, str, list[dict[str, Any]]]:
    """Ask a question with RAG context. Returns (answer, provider_label, sources_list)."""
    from src.chat.context import build_context

    sources_list: list[dict[str, Any]] = []

    if retriever is not None:
        context_string, sources_list = build_context(query, retriever)
        system = context_string + "\n\n" + SYSTEM_PROMPT
    else:
        system = SYSTEM_PROMPT

    messages = list(history or [])
    messages.append({"role": "user", "content": query})
    answer, provider_label = chat_with_fallback(
        messages, system=system, chat_provider=chat_provider
    )
    return answer, provider_label, sources_list


def create_chatbot(
    retriever: Callable[[str], list[dict[str, Any]]],
) -> Callable[..., tuple[str, str, list[dict[str, Any]]]]:
    """Return a callable that answers questions using RAG retrieval."""

    def _ask(
        query: str,
        history: list[dict[str, str]] | None = None,
        chat_provider: str | None = None,
    ) -> tuple[str, str, list[dict[str, Any]]]:
        return ask_with_rag(
            query,
            history=history,
            retriever=retriever,
            chat_provider=chat_provider,
        )

    return _ask
