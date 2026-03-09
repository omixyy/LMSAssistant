from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Parser(ABC):
    """
    Абстрактный базовый класс для всех парсеров документов
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Инициализация парсера

        Args:
            config: Словарь с настройками парсера
                   - extract_images: извлекать ли изображения
                   - detect_tables: обнаруживать ли таблицы
                   - detect_formulas: обнаруживать ли формулы
        """
        self.config = config or {}
        self._validate_config()

    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """Список поддерживаемых расширений файлов"""
        pass

    @property
    @abstractmethod
    def parser_name(self) -> str:
        """Название парсера"""
        pass

    def _validate_config(self) -> None:
        """Валидация конфигурации"""
        self.extract_images = self.config.get('extract_images', True)
        self.detect_tables = self.config.get('detect_tables', True)
        self.detect_formulas = self.config.get('detect_formulas', True)

    @abstractmethod
    def parse(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """
        Основной метод парсинга документа

        Args:
            file_path: Путь к файлу
            **kwargs: Дополнительные параметры (output_dir и др.)

        Returns:
            Dict с ключами:
                - text: извлечённый текст
                - formulas: список объектов Formula
                - tables: список объектов Table
                - images: список объектов Image
                - metadata: словарь с метаданными
        """
        pass

    def extract_text_only(self, file_path: str) -> str:
        """
        Извлечение только текста (без анализа формул/таблиц)
        Может быть переопределён в наследниках для оптимизации
        """
        result = self.parse(file_path, extract_text_only=True)
        return result.get('text', '')

    def can_parse(self, file_path: str) -> bool:
        """Проверка, может ли парсер обработать файл"""
        ext = Path(file_path).suffix.lower()
        return ext in self.supported_extensions

    def _init_result_dict(self, filename: str) -> Dict:
        """Инициализация словаря результата"""
        return {
            'text': '',
            'formulas': [],
            'tables': [],
            'images': [],
            'metadata': {
                'filename': filename,
                'pages': 0,
                'formulas_count': 0,
                'tables_count': 0,
                'images_count': 0,
                'parser': self.parser_name
            }
        }

    def _log_parse_start(self, file_path: str) -> None:
        """Логирование начала парсинга"""
        logger.info(f"[{self.parser_name}] Начинаю парсинг: {file_path}")

    def _log_parse_complete(self, result: Dict) -> None:
        """Логирование завершения парсинга"""
        logger.info(
            f"[{self.parser_name}] Парсинг завершён. "
            f"Страниц: {result['metadata'].get('pages', 0)}, "
            f"Формул: {result['metadata'].get('formulas_count', 0)}, "
            f"Таблиц: {result['metadata'].get('tables_count', 0)}, "
            f"Изображений: {result['metadata'].get('images_count', 0)}"
        )
