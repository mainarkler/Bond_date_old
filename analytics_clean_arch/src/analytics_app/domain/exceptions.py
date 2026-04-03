class DomainError(Exception):
    """Base domain exception."""


class InvalidInputError(DomainError):
    """Raised when a domain formula receives invalid input."""
