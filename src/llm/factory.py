import logging
import os
from pathlib import Path
from typing import Any

import yaml

from src.llm.base import LLMProvider
from src.llm.exceptions import ExtractionError, ProviderUnavailableError
from src.llm.providers.openai_compatible import HuggingFaceCompatibleProvider

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config" / "llm_providers.yaml"

_ROLE_ENV_VARS = {
    "chat": ("LLM_PROVIDER_CHAT", "LLM_PROVIDER"),
    "extraction": ("LLM_PROVIDER_EXTRACTION", "LLM_PROVIDER"),
}


def _config_path() -> Path:
    override = os.environ.get("LLM_PROVIDERS_CONFIG")
    if override:
        return Path(override)
    return _DEFAULT_CONFIG_PATH


def load_provider_registry(config_path: Path | None = None) -> dict[str, dict[str, Any]]:
    path = config_path or _config_path()
    if not path.is_file():
        raise FileNotFoundError(f"LLM provider registry not found: {path}")

    with path.open(encoding="utf-8") as handle:
        registry = yaml.safe_load(handle)

    if not isinstance(registry, dict) or not registry:
        raise ValueError(f"LLM provider registry is empty or invalid: {path}")

    return registry


def list_provider_names(config_path: Path | None = None) -> list[str]:
    return sorted(load_provider_registry(config_path).keys())


def _resolve_provider_name(role: str) -> str:
    role_vars = _ROLE_ENV_VARS.get(role)
    if role_vars is None:
        raise ValueError(
            f"Unknown LLM role '{role}'. Expected one of: {', '.join(_ROLE_ENV_VARS)}"
        )

    primary_var, fallback_var = role_vars
    provider_name = os.environ.get(primary_var) or os.environ.get(fallback_var)
    if not provider_name:
        valid = ", ".join(list_provider_names())
        raise ValueError(
            f"No provider configured for role '{role}'. "
            f"Set {primary_var} or {fallback_var}. "
            f"Registered providers: {valid}"
        )
    return provider_name.strip()


def _resolve_provider_order(role: str) -> list[str]:
    fallback_order = os.environ.get("LLM_FALLBACK_ORDER")
    if fallback_order:
        names = [name.strip() for name in fallback_order.split(",") if name.strip()]
        if not names:
            raise ValueError("LLM_FALLBACK_ORDER is set but contains no provider names")
        return names

    return [_resolve_provider_name(role)]


def _instantiate_provider(
    provider_name: str,
    registry: dict[str, dict[str, Any]],
) -> LLMProvider:
    if provider_name not in registry:
        valid = ", ".join(sorted(registry.keys()))
        raise ValueError(
            f"Provider '{provider_name}' is not registered. "
            f"Registered providers: {valid}"
        )

    entry = registry[provider_name]
    api_key_env = entry.get("api_key_env")
    api_key: str | None = None

    if api_key_env is not None:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            valid = ", ".join(sorted(registry.keys()))
            raise ValueError(
                f"Provider '{provider_name}' requires environment variable "
                f"{api_key_env}, which is not set. "
                f"Registered providers: {valid}"
            )

    return HuggingFaceCompatibleProvider(
        name=provider_name,
        base_url=entry["base_url"],
        api_key=api_key,
        model=entry["model"],
        supports_native_tools=bool(entry.get("supports_native_tools", False)),
        context_window=int(entry.get("context_window", 2048)),
    )


def get_provider(role: str, config_path: Path | None = None) -> LLMProvider:
    registry = load_provider_registry(config_path)
    provider_name = _resolve_provider_name(role)
    return _instantiate_provider(provider_name, registry)


def get_provider_chain(
    role: str,
    provider_order: list[str] | None = None,
    config_path: Path | None = None,
) -> list[LLMProvider]:
    registry = load_provider_registry(config_path)
    order = provider_order if provider_order is not None else _resolve_provider_order(role)
    providers: list[LLMProvider] = []

    for provider_name in order:
        if provider_name not in registry:
            logger.warning(
                "Skipping provider '%s': not registered (role=%s)",
                provider_name,
                role,
            )
            continue

        entry = registry[provider_name]
        api_key_env = entry.get("api_key_env")
        if api_key_env is not None and not os.environ.get(api_key_env):
            logger.warning(
                "Skipping provider '%s': %s is not set (role=%s)",
                provider_name,
                api_key_env,
                role,
            )
            continue

        providers.append(_instantiate_provider(provider_name, registry))

    if not providers:
        valid = ", ".join(sorted(registry.keys()))
        raise ValueError(
            f"No usable providers in fallback chain for role '{role}'. "
            f"Check LLM_FALLBACK_ORDER and API key env vars. "
            f"Registered providers: {valid}"
        )

    return providers


def format_provider_label(provider: LLMProvider) -> str:
    return f"{provider.name} · {provider.model}"


def build_provider_order(primary: str, include_fallback: bool = True) -> list[str]:
    """Put the selected provider first, optionally followed by LLM_FALLBACK_ORDER."""
    order = [primary.strip()]
    if not include_fallback:
        return order

    fallback_order = os.environ.get("LLM_FALLBACK_ORDER")
    if fallback_order:
        for name in fallback_order.split(","):
            name = name.strip()
            if name and name not in order:
                order.append(name)
    return order


def default_provider_for_role(role: str, config_path: Path | None = None) -> str:
    """Default sidebar selection: env role var, else first registered provider."""
    try:
        return _resolve_provider_name(role)
    except ValueError:
        names = list_provider_names(config_path)
        if not names:
            raise ValueError("No providers registered in config/llm_providers.yaml")
        return names[0]


def get_registered_model(provider_name: str, config_path: Path | None = None) -> str:
    registry = load_provider_registry(config_path)
    if provider_name not in registry:
        raise ValueError(f"Provider '{provider_name}' is not registered")
    return registry[provider_name]["model"]
