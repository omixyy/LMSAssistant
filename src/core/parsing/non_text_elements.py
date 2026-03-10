from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum


class FormulaType(Enum):
    """Тип формулы"""
    INLINE = 'inline'
    DISPLAY = 'display'
    DETECTED = 'detected'


@dataclass
class Formula:
    """Найденная формула"""
    page: int
    text: str
    latex: str  # LaTeX представление
    bbox: Tuple[float, float, float, float]
    context: str = ''
    confidence: float = 0.0  # Уверенность детекции (0.0-1.0)
    formula_type: str = FormulaType.INLINE.value

    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь"""
        return {
            'page': self.page,
            'latex': self.latex,
            'type': self.formula_type,
            'confidence': self.confidence,
            'context': self.context
        }


@dataclass
class Table:
    """Извлечённая таблица"""
    page: int
    bbox: Tuple[float, float, float, float]
    data: List[List[str]]  # Двумерный массив ячеек
    caption: str = ''
    headers: List[str] = field(default_factory=list)
    num_rows: int = 0
    num_cols: int = 0

    def __post_init__(self):
        """Автоматическое вычисление размеров"""
        if self.data:
            self.num_rows = len(self.data)
            self.num_cols = len(self.data[0]) if self.data else 0

    def to_markdown(self) -> str:
        """Конвертация в Markdown формат"""
        if not self.data:
            return ''

        lines = []

        # Заголовок таблицы
        if self.caption:
            lines.append(f'*{self.caption}*')

        # Формируем Markdown таблицу
        header = ' | '.join(self.headers) if self.headers else ' | '.join(str(c) for c in self.data[0])
        separator = '---|' * (self.num_cols - 1) + '---'

        lines.append(f'| {header} |')
        lines.append(f'| {separator} |')

        # Данные (начиная со второй строки, если первая использована как заголовок)
        start_row = 0 if self.headers else 1
        for row in self.data[start_row:]:
            lines.append(f'| {' | '.join(str(c) for c in row)} |')

        return '\n'.join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь"""
        return {
            'page': self.page,
            'rows': self.num_rows,
            'cols': self.num_cols,
            'caption': self.caption,
            'headers': self.headers,
            'data': self.data
        }


@dataclass
class Image:
    """Извлечённое изображение"""
    page: int
    bbox: Tuple[float, float, float, float]
    image_data: Optional[bytes] = None  # Raw bytes изображения
    image_format: str = ''  # png, jpeg, etc.
    caption: str = ''
    description: str = ''
    xref: int = 0
    width: int = 0
    height: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь"""
        return {
            'page': self.page,
            'width': self.width,
            'height': self.height,
            'format': self.image_format,
            'caption': self.caption,
            'has_data': self.image_data is not None
        }
