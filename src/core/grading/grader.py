import json
from typing import Any, Dict, Optional, List, Tuple

from core.llm.ollama_client import OllamaClient
from core.prompting.prompt_builder import PromptBuilder
from core.grading.models import Rubric, GradeBand, GradingResult, CriterionScore
from core.rag.retriever import QARetriever
from core.rag.vector_store import VectorStore


class Grader:
    def __init__(
        self,
        llm_client: OllamaClient,
        prompt_builder: PromptBuilder = None,
        qa_retriever: Optional[QARetriever] = None,
        materials_store: Optional[VectorStore] = None,
        qa_top_k_global: int = 3,
        qa_top_k_per_item: int = 1,
        materials_top_k: int = 3,
        embed_query_answer_chars: int = 2000,
        max_chars_per_rag_chunk: int = 2000,
    ) -> None:
        self._llm_client = llm_client
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._qa_retriever = qa_retriever
        self._materials_store = materials_store
        self._qa_top_k_global = qa_top_k_global
        self._qa_top_k_per_item = qa_top_k_per_item
        self._materials_top_k = materials_top_k
        self._embed_query_answer_chars = embed_query_answer_chars
        self._max_chars_per_rag_chunk = max_chars_per_rag_chunk

    @property
    def prompt_builder(self):
        return self._prompt_builder
    
    @property
    def llm_client(self):
        return self._llm_client
    
    def grade(self, task: str, answer: str, rubric: Optional[Rubric] = None, use_cot: bool = True) -> GradingResult:
        rag_context = self._build_rag_context(task=task, answer=answer, rubric=rubric)
        prompt = self._prompt_builder.build_prompt(
            task,
            answer,
            rubric,
            use_cot=use_cot,
            rag_context=rag_context,
        )

        response_text = self._llm_client.generate(prompt)
        raw = self._parse_json_response(response_text)
        return self._to_grading_result(raw)

    def _build_rag_context(self, task: str, answer: str, rubric: Optional[Rubric]) -> Optional[str]:
        if self._qa_retriever is None and self._materials_store is None:
            return None

        # Запрос для поиска должен быть достаточно релевантным текущей работе,
        # но не слишком длинным (чтобы поиск/эмбеддинг были стабильными).
        query_text = (
            f"ЗАДАНИЕ:\n{task}\n\n"
            f"ОТВЕТ СТУДЕНТА:\n{(answer or '')[: self._embed_query_answer_chars]}"
        )

        parts: List[str] = []

        if self._qa_retriever is not None:
            qa_parts: List[str] = []

            if rubric is not None and getattr(rubric, "items", None):
                for item in rubric.items:
                    # В модели/пайплайне rubric_item_id иногда может соответствовать name, а не id,
                    # поэтому пробуем оба варианта.
                    candidates = [getattr(item, "id", None), getattr(item, "name", None)]
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
                        pair = found[0]
                        qa_parts.append(
                            self._format_qa_pair(pair, rubric_item_id=getattr(item, "id", None))
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
                meta = meta or {}
                doc_text = str(doc or "").strip()
                if not doc_text:
                    continue
                doc_text = doc_text[: self._max_chars_per_rag_chunk]
                prefix = ""
                if meta:
                    # Популярные ключи (не гарантировано)
                    mtype = meta.get("type")
                    lab = meta.get("lab")
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
        rubric_part = f"rubric_item_id: {rubric_item_id or getattr(q, 'related_rubric_item_id', None) or ''}"

        return (
            f"{rubric_part}\n"
            f"Q: {q_short}\n"
            f"A: {a_short}\n"
            f"{source_part}"
        )
    
    @staticmethod
    def _parse_json_response(response_text: str) -> Dict[str, Any]:
        """
        Преобразует строку ответа модели в JSON‑объект.
        На первом шаге делаем простую реализацию с попыткой
        удалить возможные обрамляющие ```json ... ``` блоки.
        """
        cleaned = response_text.strip()

        # Простая обработка случаев, когда модель оборачивает ответ в ```json ... ```
        if cleaned.startswith('```'):
            # удаляем первую строку с ```/```json и последнюю с ```
            lines = cleaned.splitlines()
            if len(lines) >= 3:
                lines = lines[1:-1]
                cleaned = '\n'.join(lines).strip()

        return json.loads(cleaned)

    @staticmethod
    def _to_grading_result(raw: Dict[str, Any]) -> GradingResult:
        """
        Преобразует сырой JSON-ответ модели в доменную модель GradingResult.
        Ожидает структуру, похожую на ту, что формирует PromptBuilder._build_response_format.
        """
        criteria_scores = []
        for item in raw.get('criteria_scores', []):
            criteria_scores.append(
                CriterionScore(
                    confidence=float(item.get('confidence')),
                    rubric_item_id=item.get('name', ''),
                    score=float(item.get('score', 0)),
                    max_score=float(item.get('max_score', 0)),
                    justification=item.get('justification', ''),
                )
            )

        total_score = float(raw.get('overall_score', 0))
        max_score = float(raw.get('max_score', 0))
        grade_str = str(raw.get('grade', '')).lower()

        # Мягкое сопоставление строковой оценки с GradeBand
        if 'отл' in grade_str:
            band = GradeBand.EXCELLENT
            
        elif 'хор' in grade_str:
            band = GradeBand.GOOD

        elif 'удовл' in grade_str:
            band = GradeBand.SATISFACTORY

        else:
            band = GradeBand.UNSATISFACTORY

        feedback = raw.get('feedback', {}) or {}

        return GradingResult(
            rubric_id='',  # можно будет подставлять ID рубрики на уровне пайплайна
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
