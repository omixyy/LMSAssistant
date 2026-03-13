import json
from typing import Any, Dict

from core.grading.models import GradingResult
from core.llm.ollama_client import OllamaClient


class Reflector:
    """
    LLM-рефлектор: вторая модель, которая анализирует результат GRADING
    и пытается найти возможные ошибки/несогласованности в оценивании.
    """

    def __init__(self, grading_result: GradingResult, client: OllamaClient) -> None:
        self._grading_result = grading_result
        self._client = client

    @property
    def grading_result(self) -> GradingResult:
        return self._grading_result

    @property
    def client(self) -> OllamaClient:
        return self._client

    def reflect(self) -> Dict[str, Any]:
        """
        Запускает LLM для самопроверки результата GRADING.

        На вход LLM подаётся исходный JSON-ответ первой модели
        (raw_model_output). LLM должна:
        - указать, какие критерии могли быть недооценены/переоценены;
        - объяснить, почему она так считает;
        - опционально предложить скорректированные оценки.

        Возвращает разобранный JSON-ответ в виде словаря.
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
        response_text = self._client.generate(prompt)
        return self._parse_json_response(response_text)

    @staticmethod
    def _parse_json_response(response_text: str) -> Dict[str, Any]:
        """
        Аккуратно парсит JSON-ответ от LLM, удаляя возможные обёртки
        ```json ... ``` и лишний текст до/после основного JSON.
        """
        cleaned = response_text.strip()

        if cleaned.startswith('```'):
            lines = cleaned.splitlines()
            if len(lines) >= 3:
                lines = lines[1:-1]
                cleaned = '\n'.join(lines).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            start = cleaned.find('{')
            end = cleaned.rfind('}')
            if start != -1 and end != -1 and end > start:
                snippet = cleaned[start : end + 1]
                try:
                    return json.loads(snippet)
                except json.JSONDecodeError as e:
                    raise ValueError('REFLECTOR: не удалось распарсить JSON-ответ LLM') from e

            raise ValueError('REFLECTOR: ответ LLM не содержит корректного JSON')