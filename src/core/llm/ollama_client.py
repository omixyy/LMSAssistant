import ollama
from typing import Any, Dict, Optional


class OllamaClient:
    """
    Обёртка над ollama.generate с поддержкой дефолтных настроек
    """

    def __init__(self, model_name: str, default_options: Optional[Dict[str, Any]] = None) -> None:
        self._model_name = model_name
        self._default_options: Dict[str, Any] = default_options or {
            "num_ctx": 64000,
            "num_predict": 2048,
            "temperature": 0.3,
        }

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def default_options(self) -> Dict[str, Any]:
        return dict[str, Any](self._default_options)

    def update_default_options(self, **options: Any) -> None:
        self._default_options.update(options)

    def _build_config(self, prompt: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        final_options = {**self._default_options, **(options or {})}
        return {
            "model": self._model_name,
            "prompt": prompt,
            "options": final_options,
        }

    def generate(self, prompt: str, **options: Any) -> str:
        config = self._build_config(prompt=prompt, options=options or None)
        response = ollama.generate(**config)
        return response["response"]

    def generate_raw(self, prompt: str, **options: Any) -> Dict[str, Any]:
        config = self._build_config(prompt=prompt, options=options or None)
        return ollama.generate(**config)
