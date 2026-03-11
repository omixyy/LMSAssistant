from chromadb.api.types import Document
from chromadb.base_types import Metadata
from chromadb.api.types import ID

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import chromadb
from chromadb.api import ClientAPI


class VectorStore:
    """
    Обёртка над ChromaDB для хранения и поиска векторизованных документов.

    Это более общий слой, чем QARetriever: сюда можно класть любые
    текстовые объекты (методички, эталонные работы, Q&A и т.п.),
    а поверх него строить тематические ретриверы.
    """

    def __init__(
        self,
        client: Optional[ClientAPI] = None,
        collection_name: str = 'documents',
    ) -> None:
        self._client: ClientAPI = client or chromadb.Client()
        self._collection = self._client.get_or_create_collection(name=collection_name)

    @property
    def collection_name(self) -> str:
        return self._collection.name

    def add_documents(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        """
        Добавить документы в векторное хранилище.

        ids и texts должны быть одинаковой длины.
        """
        if not ids or not texts:
            return

        if len(ids) != len(texts):
            raise ValueError('ids and texts must have the same length')

        self._collection.add(
            ids=list[ID](ids),
            documents=list[Document](texts),
            metadatas=list[Metadata](metadatas) if metadatas is not None else None,
        )

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Выполнить similarity search по текстовому запросу.

        Возвращает "сырой" словарь от ChromaDB с документами, метаданными и расстояниями.
        """
        if not query_text:
            return {'ids': [[]], 'documents': [[]], 'metadatas': [[]], 'distances': [[]]}

        result = self._collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where=where,
        )
        return result
