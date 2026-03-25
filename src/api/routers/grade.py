from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_grader
from api.schemas import GradeRequest, GradeResponse, FeedbackOut, CriterionScoreOut
from core.grading.grader import Grader
from core.grading.models import Rubric, RubricItem


router = APIRouter()


def _to_domain_rubric(rubric_in) -> Rubric:
    return Rubric(
        id=rubric_in.id,
        title=rubric_in.title,
        items=[
            RubricItem(
                id=item.id,
                name=item.name,
                description=item.description,
                max_score=float(item.max_score),
            )
            for item in (rubric_in.items or [])
        ],
    )


@router.post('/grade', response_model=GradeResponse)
def grade(req: GradeRequest, grader: Grader = Depends(get_grader)) -> GradeResponse:
    rubric = _to_domain_rubric(req.rubric) if req.rubric is not None else None
    result = grader.grade(
        req.task_text,
        req.answer_text,
        rubric=rubric,
        use_cot=req.use_cot,
    )

    return GradeResponse(
        total_score=result.total_score,
        max_score=result.max_score,
        grade_band=result.grade_band.value,
        criteria=[
            CriterionScoreOut(
                rubric_item_id=c.rubric_item_id,
                score=c.score,
                max_score=c.max_score,
                justification=c.justification,
                confidence=getattr(c, 'confidence', None),
            )
            for c in result.criteria
        ],
        feedback=FeedbackOut(
            strengths=result.feedback_strengths,
            weaknesses=result.feedback_weaknesses,
            recommendations=result.feedback_recommendations,
            summary=result.feedback_summary,
        ),
        raw=result.raw_model_output if isinstance(result.raw_model_output, dict) else None,
    )

