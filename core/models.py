from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
import json


@dataclass
class CriteriaScore:
    """Оценка по конкретному критерию"""
    name: str
    score: float
    max_score: float
    comment: str = ""

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "score": self.score,
            "max_score": self.max_score,
            "comment": self.comment
        }


@dataclass
class ReviewResult:
    """Результат проверки работы"""
    filename: str
    overall_score: float
    max_score: float
    criteria_scores: List[CriteriaScore]
    feedback: str
    technical_issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return {
            "filename": self.filename,
            "overall_score": self.overall_score,
            "max_score": self.max_score,
            "criteria_scores": [c.to_dict() for c in self.criteria_scores],
            "feedback": self.feedback,
            "technical_issues": self.technical_issues,
            "recommendations": self.recommendations,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "created_at": self.created_at
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def __str__(self) -> str:
        """Красивое текстовое представление"""
        lines = []
        lines.append("=" * 60)
        lines.append(f"РЕЗУЛЬТАТ ПРОВЕРКИ: {self.filename}")
        lines.append("=" * 60)
        lines.append(f"ИТОГОВАЯ ОЦЕНКА: {self.overall_score:.1f}/{self.max_score}")
        lines.append("-" * 60)

        lines.append("ОЦЕНКИ ПО КРИТЕРИЯМ:")
        for c in self.criteria_scores:
            percentage = (c.score / c.max_score * 100) if c.max_score > 0 else 0
            lines.append(f"  {c.name}: {c.score:.1f}/{c.max_score} ({percentage:.0f}%)")
            if c.comment:
                lines.append(f"    Комментарий: {c.comment}")

        if self.technical_issues:
            lines.append("-" * 60)
            lines.append("ТЕХНИЧЕСКИЕ ЗАМЕЧАНИЯ:")
            for issue in self.technical_issues:
                lines.append(f"  • {issue}")

        lines.append("-" * 60)
        lines.append("ОБРАТНАЯ СВЯЗЬ:")
        lines.append(self.feedback)

        if self.recommendations:
            lines.append("-" * 60)
            lines.append("РЕКОМЕНДАЦИИ:")
            for rec in self.recommendations:
                lines.append(f"  • {rec}")

        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class GradingRubric:
    """Рубрика оценивания"""
    name: str
    criteria: List[Dict[str, Any]]
    description: str = ""

    @classmethod
    def default_technical(cls) -> 'GradingRubric':
        """Стандартная рубрика для технических отчетов"""
        return cls(
            name="Технический отчет",
            description="Стандартные критерии для проверки технических отчетов",
            criteria=[
                {"name": "Соответствие структуре", "max_score": 20,
                 "description": "Наличие всех обязательных разделов"},
                {"name": "Полнота описания", "max_score": 25,
                 "description": "Детальность и полнота описания выполненной работы"},
                {"name": "Корректность выводов", "max_score": 25,
                 "description": "Логичность и обоснованность выводов"},
                {"name": "Оформление (ГОСТ)", "max_score": 15,
                 "description": "Соответствие требованиям оформления"},
                {"name": "Список литературы", "max_score": 15,
                 "description": "Корректность оформления и достаточность источников"}
            ]
        )

    @classmethod
    def default_lab_report(cls) -> 'GradingRubric':
        """Рубрика для лабораторных работ"""
        return cls(
            name="Лабораторная работа",
            description="Критерии для проверки лабораторных работ",
            criteria=[
                {"name": "Цель и задачи", "max_score": 10,
                 "description": "Четкость формулировки цели и задач"},
                {"name": "Теоретическая часть", "max_score": 20,
                 "description": "Полнота теоретического обоснования"},
                {"name": "Практическая реализация", "max_score": 30,
                 "description": "Качество выполнения практической части"},
                {"name": "Анализ результатов", "max_score": 25,
                 "description": "Глубина анализа полученных результатов"},
                {"name": "Выводы", "max_score": 15,
                 "description": "Соответствие выводов поставленным задачам"}
            ]
        )