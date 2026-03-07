from typing import List, Dict, Any, Optional
import json


class PromptBuilder:
    """Построитель промптов для LLM с Chain of Thought"""

    def __init__(self, system_prompt: Optional[str] = None):
        self.system_prompt = system_prompt or self._default_system_prompt()

    def _default_system_prompt(self) -> str:
        return """Ты - опытный преподаватель высшего учебного заведения, проверяющий студенческие работы. 
Твоя задача - объективно оценить работу, дать конструктивную обратную связь и помочь студенту улучшить свои навыки.

При проверке следуй этим принципам:
1. Будь объективным и справедливым
2. Отмечай как сильные стороны, так и недостатки
3. Давай конкретные рекомендации по улучшению
4. Используй конструктивную критику
5. Основывайся на предоставленных критериях оценивания"""

    def build_prompt(self,
                     text: str,
                     task: str,
                     rubric: Dict[str, Any],
                     assignment_description: Optional[str] = None,
                     chain_of_thought: bool = True) -> str:
        """
        Построение промпта с Chain of Thought
        """
        sections = []

        # Контекст задания
        if assignment_description:
            sections.append(f"ОПИСАНИЕ ЗАДАНИЯ:\n{assignment_description}\n")

        # Рубрика оценивания
        sections.append("КРИТЕРИИ ОЦЕНИВАНИЯ:")
        for criterion in rubric.get("criteria", []):
            sections.append(
                f"- {criterion['name']} (макс. {criterion['max_score']} баллов): {criterion.get('description', '')}")
        sections.append("")

        # Текст работы
        sections.append("ТЕКСТ РАБОТЫ СТУДЕНТА:")
        sections.append(text[:10000])  # Ограничение длины
        sections.append("")

        if chain_of_thought:
            sections.append(self._build_cot_instructions())
        else:
            sections.append(self._build_direct_instructions())

        # Требование к формату ответа
        sections.append(self._build_response_format_instructions(rubric))

        return "\n".join(sections)

    def _build_cot_instructions(self) -> str:
        """Инструкции с Chain of Thought"""
        return """ИНСТРУКЦИИ ПО ПРОВЕРКЕ (следуй шагам):

ШАГ 1 - Анализ структуры:
- Определи, присутствуют ли все необходимые разделы
- Оцени логическую последовательность изложения
- Выяви отсутствующие или избыточные части

ШАГ 2 - Оценка содержания:
- Проанализируй полноту раскрытия темы
- Оцени корректность использования терминов
- Проверь обоснованность утверждений

ШАГ 3 - Проверка оформления:
- Оцени соответствие требованиям оформления
- Проверь наличие и качество иллюстративных материалов
- Проанализируй список литературы

ШАГ 4 - Оценка по критериям:
- Для каждого критерия определи балл
- Запиши обоснование оценки
- Отметь, что можно улучшить

ШАГ 5 - Формирование обратной связи:
- Обобщи основные замечания
- Выдели сильные стороны работы
- Сформулируй рекомендации
"""

    def _build_direct_instructions(self) -> str:
        """Прямые инструкции без Chain of Thought"""
        return """ИНСТРУКЦИИ:
Оцени работу по каждому критерию из рубрики.
Предоставь обоснование для каждой оценки.
Дай общую обратную связь и рекомендации.
"""

    def _build_response_format_instructions(self, rubric: Dict[str, Any]) -> str:
        """Инструкции по формату ответа"""
        criteria_example = []
        for c in rubric.get("criteria", [])[:2]:  # Пример для первых двух критериев
            criteria_example.append({
                "name": c["name"],
                "score": c["max_score"] * 0.8,  # Пример 80%
                "max_score": c["max_score"],
                "comment": "Комментарий по критерию"
            })

        example = {
            "overall_score": 85,
            "max_score": 100,
            "criteria_scores": criteria_example,
            "feedback": "Общий отзыв о работе...",
            "recommendations": ["Рекомендация 1", "Рекомендация 2"],
            "strengths": ["Сильная сторона 1"],
            "weaknesses": ["Недостаток 1"]
        }

        return f"""
ФОРМАТ ОТВЕТА:
Ответ должен быть только в формате JSON со следующей структурой:
{json.dumps(example, ensure_ascii=False, indent=2)}

ВАЖНО:
- Не добавляй никакого дополнительного текста до или после JSON
- Используй реальные оценки на основе анализа текста
- Комментарии должны быть на русском языке
- Обосновывай оценки в комментариях
- Для критериев, которые невозможно оценить, ставь 0 с пояснением причины
"""

    def build_self_reflection_prompt(self, initial_review: Dict[str, Any], text: str, task: str) -> str:
        """
        Построение промпта для саморефлексии (Reflector)
        """
        prompt = f"""Ты выполняешь роль критика, который проверяет работу другого ИИ-ассистента.

ПЕРВОНАЧАЛЬНЫЙ ОТЗЫВ:
{json.dumps(initial_review, ensure_ascii=False, indent=2)}

УСЛОВИЕ:
{task}


ТЕКСТ РАБОТЫ СТУДЕНТА:
{text[:5000]}

ЗАДАЧА:
Проанализируй первоначальный отзыв и найди:
1. Пропущенные критерии или аспекты
2. Необоснованные или противоречивые оценки
3. Возможные "галлюцинации" (утверждения, не основанные на тексте)
4. Упущенные сильные стороны или недостатки

Предоставь критический анализ в формате JSON:
{{
    "missing_criteria": ["список пропущенных аспектов"],
    "questionable_scores": [
        {{
            "criterion": "название",
            "issue": "описание проблемы",
            "suggested_review": "предложение по исправлению"
        }}
    ],
    "hallucinations": [
        {{
            "claim": "утверждение из отзыва",
            "evidence_in_text": "есть ли подтверждение в тексте"
        }}
    ],
    "additional_strengths": ["дополнительные сильные стороны"],
    "additional_weaknesses": ["дополнительные недостатки"],
    "overall_assessment": "общая оценка качества первоначального отзыва"
}}
"""
        return prompt