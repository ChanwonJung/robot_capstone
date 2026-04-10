class PromptAdapter:
    """
    Converts a user instruction into a GroundingDINO-compatible noun phrase.

    GroundingDINO expects period-separated noun phrases, e.g.:
        "bottle . cup . table"

    Current implementation is rule-based (no LLM dependency).
    Replace the adapt() body later if LLM-based nounization is needed.
    """

    def adapt(self, instruction: str) -> str:
        """
        Args:
            instruction: raw user input, e.g. "bottle", "bottle, cup", "find the red cup"
        Returns:
            GroundingDINO noun phrase string, e.g. "bottle . cup"
        """
        cleaned = instruction.strip().lower()

        # comma-separated → split and join with " . "
        parts = [p.strip() for p in cleaned.split(",") if p.strip()]

        return " . ".join(parts)
