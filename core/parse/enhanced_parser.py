import fitz  # PyMuPDF
import re
import logging
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import base64
import io

from .non_text_elements import Formula, Table, ExtractedImage
from .advanced_parser import AdvancedPDFParser

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedFileProcessor:
    """Процессор файлов с пост-обработкой результатов"""
    
    def __init__(self, extract_images: bool = True, detect_tables: bool = True):
        self.parser = AdvancedPDFParser(
            extract_images=extract_images,
            detect_tables=detect_tables
        )
    
    def process(self, file_path: str, output_dir: Optional[str] = None) -> Tuple[str, Dict]:
        """
        Обрабатывает PDF и возвращает обогащённый текст + метаданные
        """
        logger.info(f"Обработка файла: {file_path}")
        
        # Парсинг
        result = self.parser.parse_pdf(file_path, output_dir=output_dir)
        
        # Формирование обогащённого текста
        enriched_parts = ["=== ИЗВЛЕЧЁННЫЙ ТЕКСТ ДОКУМЕНТА ===\n", result['text']]
        
        # Добавляем формулы
        if result['formulas']:
            enriched_parts.append("\n\n=== РАСПОЗНАННЫЕ ФОРМУЛЫ ===")
            for i, f in enumerate(result['formulas'], 1):
                enriched_parts.append(
                    f"\n[{i}] Стр.{f.page} ({f.formula_type}, conf:{f.confidence:.2f}): ${f.latex}$"
                )
                if f.context:
                    enriched_parts.append(f"    Контекст: {f.context}")
        
        # Добавляем таблицы
        if result['tables']:
            enriched_parts.append("\n\n=== ИЗВЛЕЧЁННЫЕ ТАБЛИЦЫ ===")
            for i, t in enumerate(result['tables'], 1):
                enriched_parts.append(f"\n[{i}] Стр.{t.page} ({t.num_rows}×{t.num_cols})")
                if t.caption:
                    enriched_parts.append(f"    Подпись: {t.caption}")
                # Форматируем таблицу как Markdown
                if t.data:
                    header = " | ".join(t.headers) if t.headers else " | ".join(str(c) for c in t.data[0])
                    separator = "---|" * (t.num_cols - 1) + "---"
                    enriched_parts.append(f"    | {header} |")
                    enriched_parts.append(f"    | {separator} |")
                    for row in t.data[1:]:
                        enriched_parts.append(f"    | {' | '.join(str(c) for c in row)} |")
        
        # Добавляем изображения
        if result['images']:
            enriched_parts.append("\n\n=== ИЗОБРАЖЕНИЯ ===")
            for i, img in enumerate(result['images'], 1):
                desc = f"Стр.{img.page}, {img.width}×{img.height}px, формат: {img.image_format}"
                if img.caption:
                    desc += f", подпись: {img.caption}"
                enriched_parts.append(f"[{i}] {desc}")
        
        # Метаданные
        metadata = {
            'filename': result['metadata']['filename'],
            'pages': result['metadata']['pages'],
            'formulas_count': result['metadata']['formulas_count'],
            'tables_count': result['metadata']['tables_count'],
            'images_count': result['metadata']['images_count'],
            'formulas': [{'page': f.page, 'latex': f.latex, 'type': f.formula_type} 
                        for f in result['formulas']],
            'tables': [{'page': t.page, 'rows': t.num_rows, 'cols': t.num_cols, 'caption': t.caption} 
                      for t in result['tables']],
        }
        
        return '\n'.join(enriched_parts), metadata
    
    def export_to_markdown(self, file_path: str, output_path: str) -> None:
        """Экспорт результатов в Markdown файл"""
        text, metadata = self.process(file_path)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# {metadata['filename']}\n\n")
            f.write(f"**Страниц:** {metadata['pages']} | ")
            f.write(f"**Формул:** {metadata['formulas_count']} | ")
            f.write(f"**Таблиц:** {metadata['tables_count']} | ")
            f.write(f"**Изображений:** {metadata['images_count']}\n\n")
            f.write("---\n\n")
            f.write(text)
        
        logger.info(f"Экспорт в Markdown завершён: {output_path}")