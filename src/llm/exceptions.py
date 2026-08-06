class ProviderUnavailableError(Exception):
    """Raised when a provider cannot be reached or used (network, auth, rate-limit).

    Callers should treat this as a transient provider failure and attempt fallback.
    """


class ExtractionError(Exception):
    """Raised when structured extraction fails (schema or parsing).

    This indicates a data or validation problem, not provider unavailability.
    Callers must NOT fall back to another provider when this is raised.
    """

    def __init__(self, message: str, raw: dict | None = None):
        super().__init__(message)
        self.raw = raw
