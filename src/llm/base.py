from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class LLMProvider(ABC):
    """Abstract interface for chat and structured extraction across LLM backends."""

    @property
    @abstractmethod
    def supports_native_tools(self) -> bool:
        """Whether the provider can use native tool/function calling for extraction."""

    @property
    @abstractmethod
    def context_window(self) -> int:
        """The total token context window size for the provider."""

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        """Run a chat completion and return the assistant text response."""

    @abstractmethod
    def extract_structured(
        self,
        prompt: str,
        json_schema: dict[str, Any],
        tool_name: str,
        validation_model: type[BaseModel] | None = None,
    ) -> dict[str, Any]:
        """Extract structured data matching json_schema from the given prompt."""

    @abstractmethod
    def health_check(self) -> None:
        """Perform a trivial completion to confirm the provider is reachable.

        Raises ProviderUnavailableError if the provider cannot be used.
        """
