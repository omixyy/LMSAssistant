import json
from typing import Optional

from core.grading.models import Rubric, RubricItem


class PromptBuilder:
    """
    Класс, составляющий промпт с жесткой структурой и четкими инструкциями
    """

    # Константы для системы оценивания
    GRADING_SCALE = {
        'excellent': {'min': 85, 'max': 100, 'desc': 'Отлично'},
        'good': {'min': 70, 'max': 84, 'desc': 'Хорошо'},
        'satisfactory': {'min': 50, 'max': 69, 'desc': 'Удовлетворительно'},
        'unsatisfactory': {'min': 0, 'max': 49, 'desc': 'Неудовлетворительно'}
    }

    def __init__(self, system_prompt: str = None):
        self.system_prompt = system_prompt or self._default_system_prompt()

    def build_prompt(
        self,
        task: str,
        answer: str,
        rubric: Optional[Rubric] = None,
        use_cot: bool = True,
        rag_context: Optional[str] = None,
    ) -> str:
        """
        Вернуть полный промпт.
        Опционально включает блок с рубрикой оценивания.
        """
        parts = [
            self._default_system_prompt(),
            f"""
            ОПИСАНИЕ РАБОТЫ:
            {task}

            ОТВЕТ СТУДЕНТА:
            {answer}
            """,
        ]

        if rubric is not None:
            parts.append(self._build_rubric_block(rubric))

        if rag_context:
            parts.append(
                f"""
                === RAG-КОНТЕКСТ (reference Q&A / материалы) ===
                {rag_context}
                """
            )

        if use_cot:
            parts.append(self._build_cot_instructions())

        parts.append(self._build_response_format())

        return "\n".join(parts)

    @staticmethod
    def _default_system_prompt() -> str:
        return """
        ТЫ - ЭКСПЕРТ-ПРЕПОДАВАТЕЛЬ, проверяющий студенческие работы.

        ТВОИ ОБЯЗАННОСТИ:
        1. Строго следовать инструкциям проверки
        2. Объективно оценивать работу по критериям
        3. Давать конструктивную обратную связь
        4. Отвечать ТОЛЬКО в указанном формате JSON
        
        ПРАВИЛА:
        - Не добавляй никакого текста вне JSON
        - Будь конкретным в замечаниях
        - Основывайся только на предоставленных материалах"""

    @staticmethod
    def _build_cot_instructions() -> str:
        """Инструкции Chain of Thought"""

        return """
        ИНСТРУКЦИИ ПО ПРОВЕРКЕ (ВЫПОЛНИ КАЖДЫЙ ШАГ ПО ПОРЯДКУ):
    
        === ШАГ 1: АНАЛИЗ УСЛОВИЯ ЗАДАНИЯ ===
        1.1. Прочитай условие задания полностью
        1.2. Выдели основные требования (списком)
        1.3. Определи ожидаемый результат
    
        === ШАГ 2: АНАЛИЗ РАБОТЫ СТУДЕНТА ===
        2.1. Определи, что сделал студент
        2.2. Найди все разделы работы
        2.3. Отметь использованные методы
    
        === ШАГ 3: СОПОСТАВЛЕНИЕ С ЗАДАНИЕМ ===
        3.1. Для каждого пункта задания проверь выполнение
        3.2. Отметь полностью выполненные пункты
        3.3. Отметь частично выполненные пункты
        3.4. Отметь невыполненные пункты
    
        === ШАГ 4: ПРОВЕРКА ФОРМУЛ (если есть) ===
        4.1. Проверь каждую формулу на корректность
        4.2. Проверь правильность применения
        4.3. Найди ошибки если есть
    
        === ШАГ 5: ПРОВЕРКА ГРАФИКОВ/ТАБЛИЦ (если есть) ===
        5.1. Проверь соответствие данным
        5.2. Проверь оформление (подписи осей, единицы)
        5.3. Найди ошибки если есть
    
        === ШАГ 6: ОЦЕНКА ПО КРИТЕРИЯМ ===
        6.1. Для каждого критерия из рубрики:
             - Выставь балл от 0 до max
             - Напиши обоснование
             - Оцени свою уверенность в этой оценке (число от 0 до 1)
    
        === ШАГ 7: ФОРМИРОВАНИЕ ОБРАТНОЙ СВЯЗИ ===
        7.1. Перечисли сильные стороны
        7.2. Перечисли основные недостатки
        7.3. Дай конкретные рекомендации
        7.4. Напиши общий вывод
    
        ВАЖНО: Каждый шаг должен найти отражение в соответствующем поле JSON.
        НЕ ПРОПУСКАЙ ШАГИ. Выполняй строго по порядку.
        """

    @staticmethod
    def _build_rubric_block(rubric: Rubric) -> str:
        """
        Формирует человекочитаемый блок с рубрикой оценивания
        на основе доменной модели Rubric.
        """
        lines = [
            "РУБРИКА ОЦЕНИВАНИЯ:",
            f"Название: {rubric.title}",
            "",
            "КРИТЕРИИ:",
        ]

        for idx, item in enumerate(rubric.items, start=1):
            lines.append(
                f"{idx}. {item.name} (id: {item.id}, max: {item.max_score})"
            )
            if item.description:
                lines.append(f"   Описание: {item.description}")

        return "\n".join(lines) + "\n"

    @staticmethod
    def _build_response_format() -> str:
        """Строгий формат ответа JSON"""

        # Полный пример структуры ответа
        example_response = {
            "task_analysis": {
                "summary": "Краткое описание того, что требовалось сделать",
                "key_requirements": ["требование 1", "требование 2"],
                "expected_outcome": "Ожидаемый результат"
            },
            "work_analysis": {
                "summary": "Что сделал студент",
                "sections_present": ["введение", "теория", "расчеты"],
                "methods_used": ["бинарный поиск", "расчет погрешностей"]
            },
            "task_completion": {
                "overall_match": "полное/частичное/не соответствует",
                "completed_items": [
                    {
                        "item": "пункт задания 1",
                        "status": "выполнено/частично/не выполнено",
                        "comment": "комментарий"
                    }
                ],
                "missing_items": ["чего не хватает"]
            },
            "formulas_check": [
                {
                    "formula": "формула",
                    "correct": "True / False",
                    "errors": "описание ошибки если есть"
                }
            ],
            "graphics_check": [
                {
                    "type": "график/таблица",
                    "correct": "True / False",
                    "issues": "проблемы если есть"
                }
            ],
            "criteria_scores": [
                {
                    "name": "Соответствие тематике",
                    "score": 8,
                    "max_score": 10,
                    "justification": "Почему такая оценка",
                    "confidence": 0.85
                },
                {
                    "name": "Полнота выполнения",
                    "score": 20,
                    "max_score": 25,
                    "justification": "Почему такая оценка",
                    "confidence": 0.73
                }
            ],
            "overall_score": 75,
            "max_score": 100,
            "grade": "хорошо",
            "feedback": {
                "strengths": ["сильная сторона 1", "сильная сторона 2"],
                "weaknesses": ["недостаток 1", "недостаток 2"],
                "recommendations": ["рекомендация 1", "рекомендация 2"],
                "summary": "Общий вывод о работе"
            }
        }


        return f"""
        ТРЕБОВАНИЯ К ФОРМАТУ ОТВЕТА:
        
        1. ОТВЕТ ДОЛЖЕН БЫТЬ ТОЛЬКО В ФОРМАТЕ JSON
        2. НЕ ДОБАВЛЯЙ НИКАКОГО ТЕКСТА ДО ИЛИ ПОСЛЕ JSON
        3. ИСПОЛЬЗУЙ ТОЛЬКО РУССКИЙ ЯЗЫК ДЛЯ ТЕКСТОВЫХ ПОЛЕЙ, 
           НО В КЛЮЧАХ ТВОЕГО JSON ОТВЕТА ИСПОЛЬЗУЙ АНГЛИЙСКИЕ НАЗВАНИЯ
        4. СОБЛЮДАЙ СТРУКТУРУ JSON
        
        ПРИМЕР ПРАВИЛЬНОГО ОТВЕТА:
        ```json
        {json.dumps(example_response, ensure_ascii=False, indent=2)}
        """
