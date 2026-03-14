from typing import Any, Dict, Optional, Union

from core.grading.base_grading import GradingStep
from core.grading.models import (
    GradingResult,
    RefinerResult,
    ReflectorResult,
    Rubric,
    CriterionScore,
    GradeBand,
)
from core.llm.ollama_client import OllamaClient


def _reflector_to_dict(reflector_result: Union[ReflectorResult, Dict[str, Any]]) -> Dict[str, Any]:
    """Приводит ReflectorResult или dict к dict для подстановки в промпт."""
    if isinstance(reflector_result, dict):
        return reflector_result
    if getattr(reflector_result, 'raw', None) is not None:
        return reflector_result.raw
    return {
        'issues': [
            {
                'rubric_item_id': i.rubric_item_id,
                'problem_type': i.problem_type,
                'explanation': i.explanation,
                'current_score': i.current_score,
                'current_max_score': i.current_max_score,
            }
            for i in reflector_result.issues
        ],
        'suggested_corrections': [
            {'rubric_item_id': c.rubric_item_id, 'suggested_score': c.suggested_score, 'reason': c.reason}
            for c in reflector_result.suggested_corrections
        ],
        'overall_comment': reflector_result.overall_comment,
    }


class Refiner(GradingStep):
    def __init__(
        self,
        rubric: Rubric,
        reflector_result: Union[ReflectorResult, Dict[str, Any]],
        grading_result: GradingResult,
        client: OllamaClient,
        task: Optional[str] = None,
        answer: Optional[str] = None,
    ) -> None:
        super().__init__(client)
        self._rubric = rubric
        self._reflector_result = _reflector_to_dict(reflector_result)
        self._grading_result = grading_result
        self._task = task
        self._answer = answer

    @property
    def rubric(self) -> Rubric:
        return self._rubric

    @property
    def reflector_result(self) -> Dict[str, Any]:
        return self._reflector_result

    @property
    def task(self) -> Optional[str]:
        return self._task

    @property
    def answer(self) -> Optional[str]:
        return self._answer

    @property
    def grading_result(self) -> GradingResult:
        return self._grading_result

    def refine(self) -> RefinerResult:
        """Запускает LLM для финальной оценки, возвращает RefinerResult."""
        prompt = f"""
        Ты — третья проверяющая модель (REFINER).
        Первая модель (GRADER) уже проверила студенческую работу и выдала результат в формате JSON.
        Вторая модель (REFLECTOR) проанализировала этот результат и указала возможные ошибки.

        ТВОЯ ЗАДАЧА:
        - на основе условия задания, ответа студента, результата GRADER и вывода REFLECTOR
        сформировать финальную, сбалансированную оценку и подробный отзыв;
        - скорректировать оценки по критериям, если это обосновано;
        - явно отразить итоговые сильные и слабые стороны работы.

        ОБЯЗАТЕЛЬНО:
        - опирайся только на предоставленную информацию;
        - не придумывай факты, которых там нет;
        - верни ответ ТОЛЬКО в формате JSON и НИЧЕГО больше;
        - соблюдай приведённую ниже структуру JSON.

        УСЛОВИЕ ЗАДАНИЯ:
        \"\"\"
        {self._task or ""}
        \"\"\"

        ОТВЕТ СТУДЕНТА:
        \"\"\"
        {self._answer or ""}
        \"\"\"

        РУБРИКА:
        {self._rubric}

        РЕЗУЛЬТАТ ПЕРВОЙ МОДЕЛИ (GRADER), JSON:
        {self._grading_result.raw_model_output}

        РЕЗУЛЬТАТ ВТОРОЙ МОДЕЛИ (REFLECTOR), JSON:
        {self._reflector_result}

        ФОРМАТ ОТВЕТА (ПРИМЕР СТРУКТУРЫ, НЕ ИСПОЛЬЗУЙ КОНКРЕТНЫЕ ЧИСЛА ИЗ ПРИМЕРА):
        {{
        "refined_criteria_scores": [
            {{
            "rubric_item_id": "measurement_quality",
            "name": "Качество измерений",
            "score": 18,
            "max_score": 20,
            "justification": "Обновлённое, итоговое обоснование оценки по этому критерию"
            }}
        ],
        "refined_overall_score": 76,
        "refined_max_score": 100,
        "refined_grade": "good",
        "final_feedback": {{
            "strengths": ["..."],
            "weaknesses": ["..."],
            "recommendations": ["..."],
            "summary": "Краткий итоговый вердикт по работе"
        }}
        }}
        """
        raw = self._run_llm(prompt)
        return self.map_result(raw)

    def map_result(self, raw: Dict[str, Any]) -> RefinerResult:
        """Преобразует сырой JSON-ответ REFINER в RefinerResult (с GradingResult внутри)."""
        criteria_scores = []
        for item in raw.get('refined_criteria_scores', []):
            criteria_scores.append(
                CriterionScore(
                    rubric_item_id=item.get('rubric_item_id') or item.get('name', ''),
                    score=float(item.get('score', 0)),
                    max_score=float(item.get('max_score', 0)),
                    justification=str(item.get('justification', '')),
                )
            )
        total_score = float(raw.get('refined_overall_score', 0))
        max_score = float(raw.get('refined_max_score', 0))
        grade_str = str(raw.get('refined_grade', '')).lower()
        if 'excellent' in grade_str or 'отл' in grade_str:
            band = GradeBand.EXCELLENT
        elif 'good' in grade_str or 'хор' in grade_str:
            band = GradeBand.GOOD
        elif 'satisfactory' in grade_str or 'удовл' in grade_str:
            band = GradeBand.SATISFACTORY
        else:
            band = GradeBand.UNSATISFACTORY
        feedback = raw.get('final_feedback', {}) or {}
        refined_grading = GradingResult(
            rubric_id=self._grading_result.rubric_id,
            criteria=criteria_scores,
            total_score=total_score,
            max_score=max_score,
            grade_band=band,
            feedback_strengths=list(feedback.get('strengths', []) or []),
            feedback_weaknesses=list(feedback.get('weaknesses', []) or []),
            feedback_recommendations=list(feedback.get('recommendations', []) or []),
            feedback_summary=feedback.get('summary'),
            raw_model_output=raw,
        )
        return RefinerResult(refined_grading=refined_grading, raw=raw)