"""
Custom exception classes for provider error handling.
Used throughout the provider layer to signal specific failure modes.
"""


class CreditExhaustedError(Exception):
    """Raised when a provider has run out of paid credits (HTTP 402)."""
    pass


class QuotaExceededError(Exception):
    """Raised when a provider's rate limit has been hit (HTTP 429)."""
    pass


class ProviderUnavailableError(Exception):
    """Raised when a provider cannot be reached due to network/server errors."""
    pass
