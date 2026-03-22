from typing import Any, Dict, Optional, Union, List

from core.grading.base_grading import GradingStep
from core.grading.models import (
    GradingResult,
    RefinerResult,
    ReflectorResult,
    Rubric,
    CriterionScore,
    GradeBand,
)
from core.llm.ollama_client import OllamaClient
from core.rag.retriever import QARetriever
from core.rag.vector_store import VectorStore


class Refiner(GradingStep):
    def __init__(
        self,
        rubric: Rubric,
        reflector_result: Union[ReflectorResult, Dict[str, Any]],
        grading_result: GradingResult,
        client: OllamaClient,
        task: Optional[str] = None,
        answer: Optional[str] = None,
        qa_retriever: Optional[QARetriever] = None,
        materials_store: Optional[VectorStore] = None,
        qa_top_k_global: int = 2,
        qa_top_k_per_item: int = 1,
        materials_top_k: int = 2,
        embed_query_answer_chars: int = 2000,
        max_chars_per_rag_chunk: int = 2000,
    ) -> None:
        super().__init__(client)
        self._rubric = rubric
        self._reflector_result = self._reflector_to_dict(reflector_result)
        self._grading_result = grading_result
        self._task = task
        self._answer = answer
        self._qa_retriever = qa_retriever
        self._materials_store = materials_store
        self._qa_top_k_global = qa_top_k_global
        self._qa_top_k_per_item = qa_top_k_per_item
        self._materials_top_k = materials_top_k
        self._embed_query_answer_chars = embed_query_answer_chars
        self._max_chars_per_rag_chunk = max_chars_per_rag_chunk

    @property
    def rubric(self) -> Rubric:
        return self._rubric

    @property
    def reflector_result(self) -> Dict[str, Any]:
        return self._reflector_result

    @property
    def task(self) -> Optional[str]:
        return self._task

    @property
    def answer(self) -> Optional[str]:
        return self._answer

    @property
    def grading_result(self) -> GradingResult:
        return self._grading_result

    def refine(self) -> RefinerResult:
        """Запускает LLM для финальной оценки, возвращает RefinerResult."""
        rag_context = self._build_rag_context(task=self._task or "", answer=self._answer or "", rubric=self._rubric)
        rag_block = ""
        if rag_context:
            rag_block = f"""
            ДОПОЛНИТЕЛЬНЫЙ RAG-КОНТЕКСТ (reference Q&A / материалы):
            {rag_context}

            """

        prompt = f"""
        Ты — третья проверяющая модель (REFINER).
        Первая модель (GRADER) уже проверила студенческую работу и выдала результат в формате JSON.
        Вторая модель (REFLECTOR) проанализировала этот результат и указала возможные ошибки.

        {rag_block}

        ТВОЯ ЗАДАЧА:
        - на основе условия задания, ответа студента, результата GRADER и вывода REFLECTOR
        сформировать финальную, сбалансированную оценку и подробный отзыв;
        - скорректировать оценки по критериям, если это обосновано;
        - явно отразить итоговые сильные и слабые стороны работы.

        ОБЯЗАТЕЛЬНО:
        - опирайся только на предоставленную информацию;
        - не придумывай факты, которых там нет;
        - верни ответ ТОЛЬКО в формате JSON и НИЧЕГО больше;
        - соблюдай приведённую ниже структуру JSON.

        УСЛОВИЕ ЗАДАНИЯ:
        \"\"\"
        {self._task or ""}
        \"\"\"

        ОТВЕТ СТУДЕНТА:
        \"\"\"
        {self._answer or ""}
        \"\"\"

        РУБРИКА:
        {self._rubric}

        РЕЗУЛЬТАТ ПЕРВОЙ МОДЕЛИ (GRADER), JSON:
        {self._grading_result.raw_model_output}

        РЕЗУЛЬТАТ ВТОРОЙ МОДЕЛИ (REFLECTOR), JSON:
        {self._reflector_result}

        ФОРМАТ ОТВЕТА (ПРИМЕР СТРУКТУРЫ, НЕ ИСПОЛЬЗУЙ КОНКРЕТНЫЕ ЧИСЛА ИЗ ПРИМЕРА):
        {{
        "refined_criteria_scores": [
            {{
            "rubric_item_id": "measurement_quality",
            "name": "Качество измерений",
            "score": 18,
            "max_score": 20,
            "justification": "Обновлённое, итоговое обоснование оценки по этому критерию"
            }}
        ],
        "refined_overall_score": 76,
        "refined_max_score": 100,
        "refined_grade": "good",
        "final_feedback": {{
            "strengths": ["..."],
            "weaknesses": ["..."],
            "recommendations": ["..."],
            "summary": "Краткий итоговый вердикт по работе"
        }}
        }}
        """
        raw = self._run_llm(prompt)
        return self.map_result(raw)

    def _build_rag_context(self, task: str, answer: str, rubric: Optional[Rubric]) -> Optional[str]:
        if self._qa_retriever is None and self._materials_store is None:
            return None

        query_text = (
            f"ЗАДАНИЕ:\n{task}\n\n"
            f"ОТВЕТ СТУДЕНТА:\n{(answer or '')[: self._embed_query_answer_chars]}"
        )

        parts: List[str] = []

        if self._qa_retriever is not None:
            qa_parts: List[str] = []

            if rubric is not None and getattr(rubric, "items", None):
                for item in rubric.items:
                    candidates = [getattr(item, "id", None), getattr(item, "name", None)]
                    # дедупликация в порядке
                    seen: set[str] = set()
                    candidates = [c for c in candidates if c and not (c in seen or seen.add(c))]

                    found: List[Any] = []
                    for cand in candidates:
                        found = self._qa_retriever.query(
                            query_text=query_text,
                            top_k=self._qa_top_k_per_item,
                            rubric_item_id=cand,
                        )
                        if found:
                            break

                    if found:
                        qa_parts.append(
                            self._format_qa_pair(found[0], rubric_item_id=getattr(item, "id", None))
                        )

            if not qa_parts:
                pairs = self._qa_retriever.query(
                    query_text=query_text,
                    top_k=self._qa_top_k_global,
                    rubric_item_id=None,
                )
                for pair in pairs:
                    qa_parts.append(self._format_qa_pair(pair, rubric_item_id=None))

            if qa_parts:
                parts.append("=== Q&A REFERENCES ===\n" + "\n\n".join(qa_parts))

        if self._materials_store is not None:
            store_res = self._materials_store.query(query_text=query_text, top_k=self._materials_top_k)
            documents = store_res.get("documents", [[]])[0] or []
            metadatas = store_res.get("metadatas", [[]])[0] or []

            material_chunks: List[str] = []
            for doc, meta in zip(documents, metadatas):
                doc_text = str(doc or "").strip()
                if not doc_text:
                    continue
                doc_text = doc_text[: self._max_chars_per_rag_chunk]

                meta = meta or {}
                mtype = meta.get("type")
                lab = meta.get("lab")
                prefix = ""
                if mtype or lab:
                    prefix = f"[{mtype or 'material'} / lab={lab or '?'}] "

                material_chunks.append(prefix + doc_text)

            if material_chunks:
                parts.append("=== MATERIALS REFERENCES ===\n" + "\n\n".join(material_chunks))

        return "\n\n".join(parts) if parts else None

    def _format_qa_pair(self, pair: Any, rubric_item_id: Optional[str]) -> str:
        q = getattr(pair, "question", None)
        a = getattr(pair, "answer", None)
        question_text = str(getattr(q, "text", "") or "").strip()
        answer_text = str(a or "").strip()

        q_short = question_text[: self._max_chars_per_rag_chunk]
        a_short = answer_text[: self._max_chars_per_rag_chunk]

        source = getattr(pair, "source", None) or ""
        source_part = f"source: {source}" if source else "source: (none)"
        rubric_part = (
            f"rubric_item_id: {rubric_item_id or getattr(q, 'related_rubric_item_id', None) or ''}"
        )

        return (
            f"{rubric_part}\n"
            f"Q: {q_short}\n"
            f"A: {a_short}\n"
            f"{source_part}"
        )

    def map_result(self, raw: Dict[str, Any]) -> RefinerResult:
        """Преобразует сырой JSON-ответ REFINER в RefinerResult (с GradingResult внутри)."""
        criteria_scores = []
        for item in raw.get('refined_criteria_scores', []):
            criteria_scores.append(
                CriterionScore(
                    rubric_item_id=item.get('rubric_item_id') or item.get('name', ''),
                    score=float(item.get('score', 0)),
                    max_score=float(item.get('max_score', 0)),
                    justification=str(item.get('justification', '')),
                )
            )
        total_score = float(raw.get('refined_overall_score', 0))
        max_score = float(raw.get('refined_max_score', 0))
        grade_str = str(raw.get('refined_grade', '')).lower()
        if 'excellent' in grade_str or 'отл' in grade_str:
            band = GradeBand.EXCELLENT
        elif 'good' in grade_str or 'хор' in grade_str:
            band = GradeBand.GOOD
        elif 'satisfactory' in grade_str or 'удовл' in grade_str:
            band = GradeBand.SATISFACTORY
        else:
            band = GradeBand.UNSATISFACTORY
        feedback = raw.get('final_feedback', {}) or {}
        refined_grading = GradingResult(
            rubric_id=self._grading_result.rubric_id,
            criteria=criteria_scores,
            total_score=total_score,
            max_score=max_score,
            grade_band=band,
            feedback_strengths=list(feedback.get('strengths', []) or []),
            feedback_weaknesses=list(feedback.get('weaknesses', []) or []),
            feedback_recommendations=list(feedback.get('recommendations', []) or []),
            feedback_summary=feedback.get('summary'),
            raw_model_output=raw,
        )
        return RefinerResult(refined_grading=refined_grading, raw=raw)
    
    @staticmethod
    def _reflector_to_dict(reflector_result: Union[ReflectorResult, Dict[str, Any]]) -> Dict[str, Any]:
        """Приводит ReflectorResult или dict к dict для подстановки в промпт."""
        if isinstance(reflector_result, dict):
            return reflector_result
        if getattr(reflector_result, 'raw', None) is not None:
            return reflector_result.raw
        return {
            'issues': [
                {
                    'rubric_item_id': i.rubric_item_id,
                    'problem_type': i.problem_type,
                    'explanation': i.explanation,
                    'current_score': i.current_score,
                    'current_max_score': i.current_max_score,
                }
                for i in reflector_result.issues
            ],
            'suggested_corrections': [
                {'rubric_item_id': c.rubric_item_id, 'suggested_score': c.suggested_score, 'reason': c.reason}
                for c in reflector_result.suggested_corrections
            ],
            'overall_comment': reflector_result.overall_comment,
        }