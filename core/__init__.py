from .file_processor import FileProcessor
from .llm_analyzer import LLMAnalyzer
from .technical_validator import TechnicalValidator
from .prompt_builder import PromptBuilder
from .models import ReviewResult, CriteriaScore, GradingRubric
from typing import Optional


class ReportReviewer:
    """Основной класс системы проверки отчетов"""

    def __init__(
            self,
            model_name: str = "llama3",
            rubric: Optional[GradingRubric] = None
    ):

        self.file_processor = FileProcessor()
        self.technical_validator = TechnicalValidator()
        self.prompt_builder = PromptBuilder()
        self.llm_analyzer = LLMAnalyzer(model_name=model_name)

        self.rubric = rubric or GradingRubric.default_technical()

    def review(
            self,
            file_path: str,
            task_path: str,
            assignment_description: Optional[str] = None,
            use_reflection: bool = True
    ) -> ReviewResult:
        """
        Полный цикл проверки работы
        """
        # 1. Обработка файла
        text, metadata = self.file_processor.process(file_path)

        task, _ = self.file_processor.process(task_path)

        # 2. Техническая проверка
        technical_issues = self.technical_validator.validate(text, metadata)

        # 3. Анализ с помощью LLM
        if use_reflection:
            analysis = self.llm_analyzer.analyze_with_reflection(
                text,
                task,
                self.rubric.__dict__,
                self.prompt_builder,
            )
        else:
            prompt = self.prompt_builder.build_prompt(
                text,
                task,
                self.rubric.__dict__,
                assignment_description
            )
            analysis = self.llm_analyzer.analyze(prompt)

        # 4. Формирование результата
        criteria_scores = []
        for c in analysis.get("criteria_scores", []):
            criteria_scores.append(CriteriaScore(
                name=c.get("name", "Критерий"),
                score=float(c.get("score", 0)),
                max_score=float(c.get("max_score", 0)),
                comment=c.get("comment", "")
            ))

        result = ReviewResult(
            filename=metadata["filename"],
            overall_score=float(analysis.get("overall_score", 0)),
            max_score=float(analysis.get("max_score", 100)),
            criteria_scores=criteria_scores,
            feedback=analysis.get("feedback", ""),
            technical_issues=technical_issues,
            recommendations=analysis.get("recommendations", []),
            strengths=analysis.get("strengths", []),
            weaknesses=analysis.get("weaknesses", [])
        )

        return result


__all__ = [
    'ReportReviewer',
    'FileProcessor',
    'LLMAnalyzer',
    'TechnicalValidator',
    'PromptBuilder',
    'ReviewResult',
    'CriteriaScore',
    'GradingRubric'
]