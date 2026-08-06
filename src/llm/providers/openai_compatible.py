import json
import re
from typing import Any

from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError
from pydantic import BaseModel, ValidationError

from src.llm.base import LLMProvider
from src.llm.exceptions import ExtractionError, ProviderUnavailableError

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


class HuggingFaceCompatibleProvider(LLMProvider):
    """OpenAI-compatible provider backed by huggingface_hub.InferenceClient."""

    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str | None,
        model: str,
        supports_native_tools: bool,
        context_window: int,
    ) -> None:
        self.name = name
        self.base_url = base_url
        self.model = model
        self._supports_native_tools = supports_native_tools
        self._context_window = context_window

        client_kwargs: dict[str, Any] = {"base_url": base_url}
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        self._client = InferenceClient(**client_kwargs)

    @property
    def supports_native_tools(self) -> bool:
        return self._supports_native_tools

    @property
    def context_window(self) -> int:
        return self._context_window

    def chat(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        payload = self._build_messages(messages, system)
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=payload,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            if content is None:
                raise ProviderUnavailableError(
                    f"{self.name}: empty response content from chat completion"
                )
            return content
        except ExtractionError:
            raise
        except HfHubHTTPError as exc:
            raise self._http_error(exc) from exc
        except Exception as exc:
            raise ProviderUnavailableError(
                f"{self.name}: chat request failed ({exc.__class__.__name__}): {exc}"
            ) from exc

    def extract_structured(
        self,
        prompt: str,
        json_schema: dict[str, Any],
        tool_name: str,
        validation_model: type[BaseModel] | None = None,
    ) -> dict[str, Any]:
        if self.supports_native_tools:
            parsed = self._extract_with_native_tools(prompt, json_schema, tool_name)
        else:
            parsed = self._extract_with_json_prompt(prompt, json_schema)

        if validation_model is not None:
            try:
                validation_model.model_validate(parsed)
            except ValidationError as exc:
                raise ExtractionError(
                    f"{self.name}: extracted data failed validation: {exc}",
                    raw=parsed,
                ) from exc

        return parsed

    def health_check(self) -> None:
        self.chat(
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )

    def _build_messages(
        self,
        messages: list[dict[str, str]],
        system: str | None,
    ) -> list[dict[str, str]]:
        payload = list(messages)
        if system:
            payload = [{"role": "system", "content": system}, *payload]
        return payload

    def _extract_with_native_tools(
        self,
        prompt: str,
        json_schema: dict[str, Any],
        tool_name: str,
    ) -> dict[str, Any]:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": f"Extract structured data using {tool_name}",
                    "parameters": json_schema,
                },
            }
        ]
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                tools=tools,
                tool_choice={"type": "function", "function": {"name": tool_name}},
            )
            message = response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None)
            if not tool_calls:
                raise ExtractionError(
                    f"{self.name}: model did not return a tool call",
                    raw=None,
                )
            arguments = tool_calls[0].function.arguments
            return json.loads(arguments)
        except ExtractionError:
            raise
        except json.JSONDecodeError as exc:
            raise ExtractionError(
                f"{self.name}: tool call arguments were not valid JSON: {exc}",
                raw={"arguments": arguments} if "arguments" in locals() else None,
            ) from exc
        except HfHubHTTPError as exc:
            raise self._http_error(exc) from exc
        except Exception as exc:
            raise ProviderUnavailableError(
                f"{self.name}: structured extraction request failed "
                f"({exc.__class__.__name__}): {exc}"
            ) from exc

    def _extract_with_json_prompt(
        self,
        prompt: str,
        json_schema: dict[str, Any],
    ) -> dict[str, Any]:
        schema_text = json.dumps(json_schema, indent=2)
        system = (
            "Respond with ONLY valid JSON matching this schema, no prose, no markdown fences.\n\n"
            f"{schema_text}"
        )
        last_error: str | None = None

        for attempt in range(2):
            user_prompt = prompt
            if last_error:
                user_prompt = f"{prompt}\n\nPrevious response was invalid JSON: {last_error}"

            try:
                raw_text = self.chat(
                    messages=[{"role": "user", "content": user_prompt}],
                    system=system,
                    temperature=0.0,
                )
                cleaned = _strip_markdown_fences(raw_text)
                return json.loads(cleaned)
            except json.JSONDecodeError as exc:
                last_error = str(exc)
            except ProviderUnavailableError:
                raise

        raise ExtractionError(
            f"{self.name}: could not parse JSON from model response after retry",
            raw={"last_error": last_error},
        )

    def _http_error(self, exc: HfHubHTTPError) -> ProviderUnavailableError:
        status = getattr(exc.response, "status_code", None)
        if status in {401, 403}:
            reason = "authentication failed"
        elif status == 429:
            reason = "rate limit exceeded"
        elif status is not None and status >= 500:
            reason = "provider server error"
        else:
            reason = "provider request failed"
        return ProviderUnavailableError(
            f"{self.name}: {reason} (HTTP {status}): {exc}"
        )


def _strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return _JSON_FENCE_RE.sub("", stripped).strip()
