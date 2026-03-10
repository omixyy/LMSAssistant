from __future__ import annotations

from typing import List, Optional

from core.grading.models import GradingResult, CriterionScore


class Inquirer:
    """
    Класс для работы с этапом INQUIRING.

    Основан на конкретном результате проверки (GradingResult)
    и использует confidence по критериям, чтобы находить зоны
    наибольшей неопределенности модели.
    """

    def __init__(self, grading_result: GradingResult, confidence_threshold: float = 0.6) -> None:
        self._result = grading_result
        self._confidence_threshold = confidence_threshold

    @property
    def grading_result(self) -> GradingResult:
        return self._result

    @property
    def confidence_threshold(self) -> float:
        return self._confidence_threshold

    def get_unconfident_criteria(self, top_k: Optional[int] = None) -> List[CriterionScore]:
        """
        Вернуть критерии, по которым модель была наименее уверена.
        Критерий считается "неуверенным", если confidence < порога
        или confidence вообще отсутствует.

        Результат отсортирован по возрастанию confidence.
        Если указан top_k, список ограничивается первыми top_k элементами.
        """
        def confidence_or_zero(cs: CriterionScore) -> float:
            return cs.confidence if cs.confidence is not None else 0.0

        unconfident = [
            cs
            for cs in self._result.criteria
            if cs.confidence is None or cs.confidence < self._confidence_threshold
        ]

        unconfident_sorted = sorted(unconfident, key=confidence_or_zero)

        if top_k is not None and top_k > 0:
            return unconfident_sorted[:top_k]

        return unconfident_sorted
