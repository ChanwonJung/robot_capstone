class PromptAdapter:
    """
    쉼표 구분 입력 → GroundingDINO용 period-separated noun phrase 변환.
    GroundingDINO는 "bottle . cup . table" 형식을 기대함.
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
