from abc import ABC, abstractmethod
import json
from typing import Any, Dict

from core.llm.ollama_client import OllamaClient


class GradingStep(ABC):
    """
    Базовый класс этапа оценивания, инкапсулирующий общую
    работу с LLM и разбор JSON-ответа.
    """

    def __init__(self, client: OllamaClient) -> None:
        self._client = client

    @property
    def client(self) -> OllamaClient:
        return self._client

    def _run_llm(self, prompt: str) -> Dict[str, Any]:
        """
        Вызывает LLM с переданным промптом и возвращает
        распарсенный JSON-ответ.
        """
        response_text = self._client.generate(prompt)
        return self._parse_json_response(response_text)

    @staticmethod
    def _parse_json_response(response_text: str) -> Dict[str, Any]:
        """
        Преобразует строку ответа модели в JSON‑объект.
        Пытается аккуратно удалить возможные обрамляющие ```json ... ``` блоки
        и лишний текст до/после основного JSON.
        """
        cleaned = response_text.strip()

        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if len(lines) >= 3:
                lines = lines[1:-1]
                cleaned = "\n".join(lines).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                snippet = cleaned[start : end + 1]
                return json.loads(snippet)

            raise ValueError("LLM вернул невалидный JSON-ответ")

    @abstractmethod
    def map_result(self, raw: Dict[str, Any]) -> Any:
        """
        Преобразует сырой JSON-ответ этапа в доменную модель.
        """
        raise NotImplementedError
