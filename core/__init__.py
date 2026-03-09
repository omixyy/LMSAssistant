"""
Основной пакет системы проверки студенческих работ
"""

from .parser import (
    Parser, PDFParser, DOCXParser, ParserFactory,
    Formula, Table, Image, FormulaType
)
from .processor import DocumentProcessor

__all__ = [
    # Парсеры
    'Parser',
    'PDFParser',
    'DOCXParser',
    'ParserFactory',

    # Элементы
    'Formula',
    'Table',
    'Image',
    'FormulaType',

    # Процессор
    'DocumentProcessor',
]