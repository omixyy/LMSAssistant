from __future__ import annotations

from chromadb.api.types import Document
from chromadb.base_types import Metadata
from typing import List, Sequence, Optional

from chromadb.api import ClientAPI

from core.grading.models import QAPair, Question


class QARetriever:
    """
    Обёртка над ChromaDB для хранения и поиска Q&A-пар.

    Отвечает за:
    - добавление новых QAPair в коллекцию;
    - поиск релевантных пар по текстовому запросу.
    """

    def __init__(self, client: ClientAPI, collection_name: str = 'qa_pairs') -> None:
        self._client = client
        self._collection = client.get_or_create_collection(name=collection_name)

    @property
    def collection_name(self) -> str:
        return self._collection.name

    def add_pairs(self, pairs: Sequence[QAPair]) -> None:
        """
        Добавить Q&A-пары в коллекцию.

        В качестве документа используется конкатенация текста вопроса и ответа,
        а также текст вопроса как метаданные для фильтрации/отображения.
        """
        if not pairs:
            return

        ids = []
        documents = []
        metadatas = []

        for idx, pair in enumerate[QAPair](pairs):
            q: Question = pair.question
            qa_id = f'{q.id}-{idx}'
            ids.append(qa_id)
            documents.append(f'Q: {q.text}\nA: {pair.answer}')
            metadatas.append(
                {
                    'question_id': q.id,
                    'question_text': q.text,
                    'rubric_item_id': q.related_rubric_item_id or '',
                    'source': pair.source or '',
                }
            )

        self._collection.add(ids=ids, documents=documents, metadatas=metadatas)

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        rubric_item_id: Optional[str] = None,
    ) -> List[QAPair]:
        """
        Найти top_k релевантных Q&A-пар по текстовому запросу.

        При наличии rubric_item_id дополнительно фильтрует результаты
        по связанному критерию.
        """
        if not query_text:
            return []

        where = None
        if rubric_item_id:
            where = {'rubric_item_id': rubric_item_id}

        result = self._collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where=where,
        )

        documents = result.get('documents', [[]])[0]
        metadatas = result.get('metadatas', [[]])[0]

        pairs: List[QAPair] = []
        for doc, meta in zip(documents, metadatas):
            question_text = meta.get('question_text', '')
            question_id = meta.get('question_id', '')
            related_rubric_item_id = meta.get('rubric_item_id') or None

            question = Question(
                id=question_id,
                text=question_text,
                related_rubric_item_id=related_rubric_item_id,
            )
            # Ответ извлекаем из полного документа 'Q: ...\nA: ...'
            answer = doc.split('\nA:', 1)[-1].strip() if '\nA:' in doc else doc

            pairs.append(
                QAPair(
                    question=question,
                    answer=answer,
                    source=meta.get('source') or None,
                )
            )

        return pairs
