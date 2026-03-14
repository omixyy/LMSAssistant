import json
from typing import Any, Dict

from core.grading.base_grading import GradingStep
from core.grading.models import (
    GradingResult,
    ReflectorResult,
    ReflectorIssue,
    ReflectorCorrection,
)
from core.llm.ollama_client import OllamaClient


class Reflector(GradingStep):
    """
    LLM-рефлектор: вторая модель, которая анализирует результат GRADING
    и пытается найти возможные ошибки/несогласованности в оценивании.
    """

    def __init__(self, grading_result: GradingResult, client: OllamaClient) -> None:
        super().__init__(client)
        self._grading_result = grading_result

    @property
    def grading_result(self) -> GradingResult:
        return self._grading_result

    def reflect(self) -> ReflectorResult:
        """
        Запускает LLM для самопроверки результата GRADING,
        возвращает доменную модель ReflectorResult.
        """
        if self._grading_result.raw_model_output is None:
            raise ValueError(
                'GradingResult.raw_model_output отсутствует, нечего рефлексировать.'
            )

        grading_json = json.dumps(
            self._grading_result.raw_model_output,
            ensure_ascii=False,
            indent=2,
        )

        prompt = f"""
        Ты — вторая проверяющая модель (REFLECTOR).
        Первая модель уже проверила студенческую работу и выдала результат в формате JSON.

        ТВОЯ ЗАДАЧА:
        - критически проанализировать этот результат;
        - найти возможные ошибки, несогласованности или перекосы в оценках по критериям;
        - указать, какие критерии могли быть НЕДООЦЕНЕНЫ или ПЕРЕОЦЕНЕНЫ;
        - при необходимости предложить скорректированные оценки и кратко объяснить почему.

        ОБЯЗАТЕЛЬНО:
        - опирайся только на предоставленный JSON;
        - не придумывай факты, которых там нет;
        - верни ответ ТОЛЬКО в формате JSON и НИЧЕГО больше.

        ФОРМАТ ОТВЕТА:
        {{
        "issues": [
            {{
            "rubric_item_id": "string",
            "problem_type": "underestimated / overestimated / inconsistent / unclear",
            "explanation": "почему это выглядит проблемным",
            "current_score": 0,
            "current_max_score": 0
            }}
        ],
        "suggested_corrections": [
            {{
            "rubric_item_id": "string",
            "suggested_score": 0,
            "reason": "почему стоит изменить оценку"
            }}
        ],
        "overall_comment": "краткий вывод о качестве исходного оценивания"
        }}

        ВХОДНОЙ JSON РЕЗУЛЬТАТА GRADING:
        ```json
        {grading_json}
        ```
        """
        raw = self._run_llm(prompt)
        return self.map_result(raw)

    def map_result(self, raw: Dict[str, Any]) -> ReflectorResult:
        """Преобразует сырой JSON-ответ REFLECTOR в ReflectorResult."""
        issues = []
        for item in raw.get('issues', []):
            issues.append(
                ReflectorIssue(
                    rubric_item_id=str(item.get('rubric_item_id', '')),
                    problem_type=str(item.get('problem_type', '')),
                    explanation=str(item.get('explanation', '')),
                    current_score=float(item.get('current_score', 0)),
                    current_max_score=float(item.get('current_max_score', 0)),
                )
            )
        corrections = []
        for item in raw.get('suggested_corrections', []):
            corrections.append(
                ReflectorCorrection(
                    rubric_item_id=str(item.get('rubric_item_id', '')),
                    suggested_score=float(item.get('suggested_score', 0)),
                    reason=str(item.get('reason', '')),
                )
            )
        return ReflectorResult(
            issues=issues,
            suggested_corrections=corrections,
            overall_comment=str(raw.get('overall_comment', '')),
            raw=raw,
        )