from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Any, Optional


@dataclass
class Formula:
    """Найденная формула"""
    page: int
    text: str
    latex: str  # LaTeX представление (если удалось распознать)
    bbox: Tuple[float, float, float, float]
    context: str = ""
    confidence: float = 0.0  # Уверенность детекции (0.0-1.0)
    formula_type: str = "inline"  # inline, display, detected


@dataclass
class Table:
    """Извлечённая таблица"""
    page: int
    bbox: Tuple[float, float, float, float]
    data: List[List[str]]  # Двумерный массив ячеек
    caption: str = ""
    headers: List[str] = field(default_factory=list)
    num_rows: int = 0
    num_cols: int = 0


@dataclass
class ExtractedImage:
    """Извлечённое изображение"""
    page: int
    bbox: Tuple[float, float, float, float]
    image_data: Optional[bytes] = None  # Raw bytes изображения
    image_format: str = ""  # png, jpeg, etc.
    caption: str = ""
    description: str = ""
    xref: int = 0
    width: int = 0
    height: int = 0