"""
Пакет для парсинга документов
"""

from .base_parser import Parser
from .pdf_parser import PDFParser
from .docx_parser import DOCXParser
from .factory import ParserFactory
from .non_text_elements import Formula, Table, Image, FormulaType

__all__ = [
    'Parser',
    'PDFParser',
    'DOCXParser',
    'ParserFactory',
    'Formula',
    'Table',
    'Image',
    'FormulaType'
]
