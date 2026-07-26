"""Token estimation and usage tracking service."""

from __future__ import annotations

from dataclasses import dataclass

from config import SETTINGS
from utils.metrics import estimate_tokens


@dataclass
class TokenUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int


class TokenTracker:
    """Estimates token usage for prompts and responses using a character heuristic."""

    def __init__(self, chars_per_token: float | None = None) -> None:
        self.chars_per_token = chars_per_token or SETTINGS.chars_per_token_estimate

    def estimate(self, input_text: str, output_text: str) -> TokenUsage:
        """Estimate input, output, and total token counts for a request/response pair."""
        input_tokens = estimate_tokens(input_text, self.chars_per_token)
        output_tokens = estimate_tokens(output_text, self.chars_per_token)
        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )

    def estimate_conversation(self, turns: list[tuple[str, str]]) -> TokenUsage:
        """Estimate cumulative token usage across a list of (question, answer) turns."""
        total_input = 0
        total_output = 0
        for question, answer in turns:
            total_input += estimate_tokens(question, self.chars_per_token)
            total_output += estimate_tokens(answer, self.chars_per_token)
        return TokenUsage(
            input_tokens=total_input,
            output_tokens=total_output,
            total_tokens=total_input + total_output,
        )
