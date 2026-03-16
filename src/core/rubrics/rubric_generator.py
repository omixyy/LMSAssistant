from __future__ import annotations

import json
from typing import Any, Dict, List

from core.llm.ollama_client import OllamaClient
from core.grading.models import Rubric, RubricItem


class RubricGenerator:
    """
    Класс для полуавтоматической генерации рубрики оценивания
    на основе текста задания/методических указаний.

    Отвечает за:
    - построение промпта для LLM;
    - вызов модели через OllamaClient;
    - преобразование JSON-ответа в Rubric/RubricItem.
    """

    def __init__(self, llm_client: OllamaClient) -> None:
        self._llm_client = llm_client

    @property
    def llm_client(self) -> OllamaClient:
        return self._llm_client

    def generate_from_text(self, rubric_id: str, title: str, assignment_text: str) -> Rubric:
        """
        Сгенерировать черновик рубрики на основе текста задания.

        Возвращает доменную модель Rubric, которую преподаватель
        затем может отредактировать и сохранить.
        """
        prompt = self._build_prompt(assignment_text)
        response_text = self._llm_client.generate(prompt)
        raw = self._parse_json_response(response_text)
        items = self._to_rubric_items(raw)
        return Rubric(id=rubric_id, title=title, items=items)

    @staticmethod
    def _build_prompt(assignment_text: str) -> str:
        """
        Строит промпт для LLM, просящий выделить критерии оценивания
        и максимальные баллы в формате JSON.
        """
        return f"""
        Ты — опытный преподаватель университета.
        Тебе дан текст задания/методических указаний.
        Твоя задача — предложить черновик рубрики оценивания
        в формате JSON без какого-либо дополнительного текста.

        ТРЕБОВАНИЯ:
        - Выдели 4–8 ключевых критериев оценивания.
        - Для каждого критерия укажи:
        - machine_id — короткий идентификатор латиницей (например, "coverage", "formatting").
        - name — человекочитаемое название по-русски.
        - description — краткое текстовое описание, что именно оценивается.
        - max_score — максимальное количество баллов (число).
        - Следи, чтобы формулировки были понятны студентам и преподавателям.

        ФОРМАТ ОТВЕТА (ТОЛЬКО JSON):
        {{
        "items": [
            {{
            "machine_id": "coverage",
            "name": "Соответствие тематике",
            "description": "Насколько работа соответствует теме и требованиям задания.",
            "max_score": 10
            }}
        ]
        }}

        ТЕКСТ ЗАДАНИЯ:
        \"\"\"
        {assignment_text}
        \"\"\"
        """

    @staticmethod
    def _parse_json_response(response_text: str) -> Dict[str, Any]:
        """
        Преобразует строку ответа модели в JSON-объект,
        удаляя возможные ```json ... ``` обёртки.
        """
        cleaned = response_text.strip()

        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if len(lines) >= 3:
                lines = lines[1:-1]
                cleaned = "\n".join(lines).strip()
        print(cleaned)
        return json.loads(cleaned)

    @staticmethod
    def _to_rubric_items(raw: Dict[str, Any]) -> List[RubricItem]:
        """
        Преобразует сырой JSON в список RubricItem.
        Ожидает структуру: {{ "items": [{{ ... }}] }}.
        """
        items: List[RubricItem] = []
        for item in raw.get("items", []):
            machine_id = str(item.get("machine_id") or "").strip()
            name = str(item.get("name") or "").strip() or machine_id
            description = str(item.get("description") or "").strip()
            max_score_raw = item.get("max_score", 0)
            try:
                max_score = float(max_score_raw)
            except (TypeError, ValueError):
                max_score = 0.0

            items.append(
                RubricItem(
                    id=machine_id or name,
                    name=name,
                    description=description,
                    max_score=max_score,
                )
            )

        return items