from __future__ import annotations

from typing import List, Optional

from core.grading.models import (
    GradingResult, 
    CriterionScore, 
    Question,
)


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

    def generate_questions(self, top_k: Optional[int] = None) -> List[Question]:
        """
        Сгенерировать список уточняющих вопросов для преподавателя
        по критериям с наименьшей уверенностью модели.

        Вопросы формируются шаблонно, без использования LLM.
        """
        questions: List[Question] = []
        unconfident = self.get_unconfident_criteria(top_k=top_k)

        for idx, cs in enumerate[CriterionScore](unconfident, start=1):
            conf_str = (
                f'{cs.confidence:.2f}' if cs.confidence is not None else 'не указана'
            )
            q_id = f"q_{cs.rubric_item_id}_{idx}"
            text = (
                'Поясните, пожалуйста, как правильно трактовать критерий '
                f'«{cs.rubric_item_id}» в случаях, подобных этой работе.\n'
                f'Модель поставила {cs.score} из {cs.max_score} баллов '
                f'(уверенность: {conf_str}) со следующим обоснованием:\n'
                f'«{cs.justification}».\n'
                'Какие признаки Вы считаете ключевыми для этого критерия, '
                'и в каких ситуациях оценку следует повышать или понижать?'
            )

            questions.append(
                Question(
                    id=q_id,
                    text=text,
                    related_rubric_item_id=cs.rubric_item_id,
                )
            )

        return questions
