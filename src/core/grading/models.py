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
    id: str
    name: str
    description: str
    max_score: float


@dataclass
class Rubric:
    id: str
    title: str
    items: List[RubricItem] = field(default_factory=list)


@dataclass
class CriterionScore:
    rubric_item_id: str
    score: float
    max_score: float
    justification: str
    confidence: Optional[float] = None


@dataclass
class GradingResult:
    rubric_id: str
    criteria: List[CriterionScore]
    total_score: float
    max_score: float
    grade_band: GradeBand
    feedback_strengths: List[str] = field(default_factory=list)
    feedback_weaknesses: List[str] = field(default_factory=list)
    feedback_recommendations: List[str] = field(default_factory=list)
    feedback_summary: Optional[str] = None
    raw_model_output: Optional[Any] = None


@dataclass
class ExpertScore:
    rubric_item_id: str
    score: float


@dataclass
class ErrorSample:
    rubric_item_id: str
    model_score: float
    expert_score: float
    model_justification: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class Question:
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
    question: Question
    confidence: float


@dataclass
class QAPair:
    question: Question
    answer: str
    source: Optional[str] = None


@dataclass
class AdaptationRule:
    id: str
    rubric_item_id: Optional[str]
    description: str


@dataclass
class RubricUpdate:
    original_rubric_id: str
    updated_rubric: Rubric
    applied_rules: List[AdaptationRule] = field(default_factory=list)


@dataclass
class InquirerResult:
    questions: List[Question]
    unconfident_criteria: List[CriterionScore]
    raw: Optional[Any] = None


@dataclass
class ReflectorIssue:
    rubric_item_id: str
    problem_type: str
    explanation: str
    current_score: float
    current_max_score: float


@dataclass
class ReflectorCorrection:
    rubric_item_id: str
    suggested_score: float
    reason: str


@dataclass
class ReflectorResult:
    issues: List[ReflectorIssue]
    suggested_corrections: List[ReflectorCorrection]
    overall_comment: str
    raw: Optional[Any] = None


@dataclass
class RefinerResult:
    refined_grading: GradingResult
    raw: Optional[Any] = None

