import ollama
import json
import logging
from typing import Dict, Any, Optional
import time

logger = logging.getLogger(__name__)


class LLMAnalyzer:
    """Анализатор на основе локальной LLM через Ollama"""

    def __init__(
            self,
            model_name: str = 'gemma3:12b',
            host: str = 'http://localhost:11434',
            temperature: float = 0.3,
            timeout: int = 120,
    ):
        self.model_name = model_name
        self.host = host
        self.temperature = temperature
        self.timeout = timeout

        self._check_model()

    def _check_model(self):
        """Проверка доступности модели в Ollama"""
        try:
            models = ollama.list()
            model_names = [m['name'] for m in models.get('models', [])]

            if self.model_name not in model_names:
                logger.warning(f'Модель {self.model_name} не найдена. Доступны: {model_names}')
                logger.info(f'Попробуйте загрузить модель: ollama pull {self.model_name}')
            else:
                logger.info(f'Модель {self.model_name} доступна')

        except Exception as e:
            logger.error(f'Ошибка подключения к Ollama: {str(e)}')
            logger.info('Убедитесь, что Ollama запущена: ollama serve')

    def analyze(self, prompt: str, max_retries: int = 3) -> Dict[str, Any]:
        """Отправка промпта в LLM и получение анализа"""
        for attempt in range(max_retries):
            try:
                logger.info(f'Отправка запроса к модели {self.model_name} (попытка {attempt + 1})')

                start_time = time.time()

                response = ollama.chat(
                    model=self.model_name,
                    messages=[{'role': 'user', 'content': prompt}],
                    options={
                        'temperature': self.temperature,
                        'num_predict': 4096,
                    }
                )

                elapsed = time.time() - start_time
                logger.info(f'Ответ получен за {elapsed:.1f} сек')

                result_text = response['message']['content']
                return self._parse_response(result_text)

            except Exception as e:
                logger.error(f'Ошибка при обращении к Ollama (попытка {attempt + 1}): {str(e)}')
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)

        raise RuntimeError('Не удалось получить ответ от LLM')

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Парсинг ответа от LLM, извлечение JSON"""
        try:
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1

            if 0 <= start_idx < end_idx:
                json_str = response_text[start_idx:end_idx]
                result = json.loads(json_str)

                required_fields = ['overall_score', 'max_score', 'criteria_scores', 'feedback']
                for field in required_fields:
                    if field not in result:
                        logger.warning(f'Отсутствует обязательное поле: {field}')
                        result[field] = self._get_default_value(field)

                return result
            else:
                logger.error('JSON не найден в ответе модели')
                return self._create_fallback_response(response_text)

        except json.JSONDecodeError as e:
            logger.error(f'Ошибка парсинга JSON: {str(e)}')
            return self._create_fallback_response(response_text)

    def _get_default_value(self, field: str) -> Any:
        """Значения по умолчанию для обязательных полей"""
        defaults = {
            'overall_score': 0,
            'max_score': 100,
            'criteria_scores': [],
            'feedback': 'Не удалось распарсить ответ модели',
            'recommendations': ['Повторите попытку позже']
        }
        return defaults.get(field, '')

    def _create_fallback_response(self, raw_text: str) -> Dict[str, Any]:
        """Создание ответа по умолчанию при ошибке парсинга"""
        return {
            'overall_score': 0,
            'max_score': 100,
            'criteria_scores': [],
            'feedback': f'Ошибка обработки ответа модели. Сырой ответ: {raw_text[:500]}',
            'recommendations': ['Проверьте формат промпта', 'Попробуйте другую модель'],
            'strengths': [],
            'weaknesses': [],
            'error': True
        }

    def analyze_with_reflection(
            self,
            text: str,
            task: str,
            rubric: Dict[str, Any],
            prompt_builder,
            metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Двухэтапный анализ с саморефлексией
        Модель сама определяет тип работы по заданию
        """
        logger.info('Этап 1: Первичный анализ (модель анализирует задание и работу)')
        prompt1 = prompt_builder.build_prompt(
            text=text,
            task=task,
            rubric=rubric,
            metadata=metadata,
            chain_of_thought=True
        )
        initial_result = self.analyze(prompt1)

        logger.info('Этап 2: Саморефлексия (проверка соответствия заданию)')
        reflection_prompt = prompt_builder.build_self_reflection_prompt(
            initial_result,
            text,
            task
        )
        reflection = self.analyze(reflection_prompt)

        logger.info('Этап 3: Формирование итогового результата')

        # Добавляем информацию о рефлексии
        initial_result['_reflection'] = reflection

        # Логируем, как модель поняла задание
        if 'task_analysis' in initial_result:
            logger.info(f"Анализ задания моделью: {initial_result['task_analysis'][:200]}")

        return initial_result