"""User-facing outcomes that are not crashes."""

from __future__ import annotations


class NoPersonError(ValueError):
    """The video cannot be analysed because no person (visible torso) is found. The message is shown to the user."""

    def __init__(self, message: str, presence: dict | None = None):
        super().__init__(message)
        self.presence = presence or {}
