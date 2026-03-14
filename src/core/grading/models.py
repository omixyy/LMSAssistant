from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Any


class GradeBand(str, Enum):
    EXCELLENT = 'excellent'
    GOOD = 'good'
    SATISFACTORY = 'satisfactory'
    UNSATISFACTORY = 'unsatisfactory'


@dataclass
class RubricItem:
    """Отдельный критерий рубрики."""

    id: str
    name: str
    description: str
    max_score: float


@dataclass
class Rubric:
    """Полная рубрика оценивания."""

    id: str
    title: str
    items: List[RubricItem] = field(default_factory=list)


@dataclass
class CriterionScore:
    """Оценка по одному критерию."""

    rubric_item_id: str
    score: float
    max_score: float
    justification: str
    confidence: Optional[float] = None


@dataclass
class GradingResult:
    """
    Результат этапа GRADING в доменных терминах,
    без привязки к формату JSON, который вернула модель.
    """

    rubric_id: str
    criteria: List[CriterionScore]
    total_score: float
    max_score: float
    grade_band: GradeBand
    feedback_strengths: List[str] = field(default_factory=list)
    feedback_weaknesses: List[str] = field(default_factory=list)
    feedback_recommendations: List[str] = field(default_factory=list)
    feedback_summary: Optional[str] = None
    raw_model_output: Optional[Any] = None  # при необходимости можно хранить сырой ответ модели


@dataclass
class ExpertScore:
    """Эталонная оценка эксперта по критерию."""

    rubric_item_id: str
    score: float


@dataclass
class ErrorSample:
    """
    Пример расхождения модели с экспертом
    для дальнейшего анализа на этапах INQUIRING/OPTIMIZING.
    """

    rubric_item_id: str
    model_score: float
    expert_score: float
    model_justification: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class Question:
    """Вопрос, сгенерированный этапом INQUIRING."""

    id: str
    text: str
    related_rubric_item_id: Optional[str] = None

    def __str__(self) -> str:
        return f"""
        ID: {self.id}
        Text: {self.text}
        Related Rubric Item ID: {self.related_rubric_item_id}
        """


@dataclass
class QuestionRank:
    """Вопрос с оценкой важности/уверенности."""

    question: Question
    confidence: float


@dataclass
class QAPair:
    """Пара вопрос–ответ, используемая для обучения и RAG."""

    question: Question
    answer: str
    source: Optional[str] = None  # например, id эксперта или сессии


@dataclass
class AdaptationRule:
    """Правило адаптации рубрики, полученное на этапе OPTIMIZING."""

    id: str
    rubric_item_id: Optional[str]
    description: str


@dataclass
class RubricUpdate:
    """Изменения рубрики на основе AdaptationRule."""

    original_rubric_id: str
    updated_rubric: Rubric
    applied_rules: List[AdaptationRule] = field(default_factory=list)


# --- Результаты этапов INQUIRING, REFLECTOR, REFINER ---


@dataclass
class InquirerResult:
    """
    Результат этапа INQUIRING: список уточняющих вопросов
    по критериям с низкой уверенностью модели.
    """

    questions: List[Question]
    unconfident_criteria: List[CriterionScore]
    raw: Optional[Any] = None


@dataclass
class ReflectorIssue:
    """Один проблемный критерий из вывода REFLECTOR."""

    rubric_item_id: str
    problem_type: str  # underestimated / overestimated / inconsistent / unclear
    explanation: str
    current_score: float
    current_max_score: float


@dataclass
class ReflectorCorrection:
    """Предложенная REFLECTOR'ом корректировка оценки по критерию."""

    rubric_item_id: str
    suggested_score: float
    reason: str


@dataclass
class ReflectorResult:
    """
    Результат этапа REFLECTOR: анализ ошибок GRADING
    и предложенные корректировки (без финальной оценки).
    """

    issues: List[ReflectorIssue]
    suggested_corrections: List[ReflectorCorrection]
    overall_comment: str
    raw: Optional[Any] = None


@dataclass
class RefinerResult:
    """
    Результат этапа REFINER: финальная оценка и отзыв
    после учёта рефлексии (без confidence по критериям).
    """

    refined_grading: GradingResult
    raw: Optional[Any] = None

