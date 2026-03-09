from typing import Optional, Dict, Type, List
from pathlib import Path

from .base_parser import Parser
from .pdf_parser import PDFParser
from .docx_parser import DOCXParser


class ParserFactory:
    """
    Фабрика для создания парсеров
    """

    _parsers: Dict[str, Type[Parser]] = {
        '.pdf': PDFParser,
        '.docx': DOCXParser,
        '.doc': DOCXParser,
    }

    @classmethod
    def register_parser(cls, extension: str, parser_class: Type[Parser]) -> None:
        """Регистрация нового парсера для расширения"""
        cls._parsers[extension.lower()] = parser_class

    @classmethod
    def create_parser(cls, file_path: str, config: Optional[Dict] = None) -> Optional[Parser]:
        """
        Создание подходящего парсера для файла

        Args:
            file_path: Путь к файлу
            config: Конфигурация для парсера

        Returns:
            Экземпляр парсера или None, если парсер не найден
        """
        ext = Path(file_path).suffix.lower()
        parser_class = cls._parsers.get(ext)

        if parser_class:
            return parser_class(config or {})

        return None

    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """Список поддерживаемых расширений"""
        return list(cls._parsers.keys())
