import json
from typing import Any, Dict, Optional

from core.llm.ollama_client import OllamaClient
from core.prompting.prompt_builder import PromptBuilder
from core.grading.models import Rubric
from core.grading.models import (
    GradeBand,
    GradingResult,
    CriterionScore,
)


class Grader:
    def __init__(self, llm_client: OllamaClient, prompt_builder: PromptBuilder = None) -> None:
        self._llm_client = llm_client
        self._prompt_builder = prompt_builder or PromptBuilder()

    @property
    def prompt_builder(self):
        return self._prompt_builder
    
    @property
    def llm_client(self):
        return self._llm_client
    
    def grade(self, task: str, answer: str, rubric: Optional[Rubric] = None, use_cot: bool = True) -> GradingResult:
        prompt = self._prompt_builder.build_prompt(
            task,
            answer,
            rubric,
            use_cot=use_cot,
        )

        response_text = self._llm_client.generate(prompt)
        raw = self._parse_json_response(response_text)
        return self._to_grading_result(raw)
    
    @staticmethod
    def _parse_json_response(response_text: str) -> Dict[str, Any]:
        """
        Преобразует строку ответа модели в JSON‑объект.
        На первом шаге делаем простую реализацию с попыткой
        удалить возможные обрамляющие ```json ... ``` блоки.
        """
        cleaned = response_text.strip()

        # Простая обработка случаев, когда модель оборачивает ответ в ```json ... ```
        if cleaned.startswith('```'):
            # удаляем первую строку с ```/```json и последнюю с ```
            lines = cleaned.splitlines()
            if len(lines) >= 3:
                lines = lines[1:-1]
                cleaned = '\n'.join(lines).strip()

        return json.loads(cleaned)

    @staticmethod
    def _to_grading_result(raw: Dict[str, Any]) -> GradingResult:
        """
        Преобразует сырой JSON-ответ модели в доменную модель GradingResult.
        Ожидает структуру, похожую на ту, что формирует PromptBuilder._build_response_format.
        """
        criteria_scores = []
        for item in raw.get('criteria_scores', []):
            criteria_scores.append(
                CriterionScore(
                    confidence=float(item.get('confidence')),
                    rubric_item_id=item.get('name', ''),
                    score=float(item.get('score', 0)),
                    max_score=float(item.get('max_score', 0)),
                    justification=item.get('justification', ''),
                )
            )

        total_score = float(raw.get('overall_score', 0))
        max_score = float(raw.get('max_score', 0))
        grade_str = str(raw.get('grade', '')).lower()

        # Мягкое сопоставление строковой оценки с GradeBand
        if 'отл' in grade_str:
            band = GradeBand.EXCELLENT
            
        elif 'хор' in grade_str:
            band = GradeBand.GOOD

        elif 'удовл' in grade_str:
            band = GradeBand.SATISFACTORY

        else:
            band = GradeBand.UNSATISFACTORY

        feedback = raw.get('feedback', {}) or {}

        return GradingResult(
            rubric_id='',  # можно будет подставлять ID рубрики на уровне пайплайна
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
