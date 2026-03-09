import logging
from typing import Optional, Dict, Tuple, List
from pathlib import Path

from .parser.factory import ParserFactory

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Процессор документов.
    Координирует парсинг и форматирование результатов для использования в LLM.
    """

    def __init__(self,
                 extract_images: bool = True,
                 detect_tables: bool = True,
                 detect_formulas: bool = True):
        """
        Инициализация процессора

        Args:
            extract_images: Извлекать ли изображения
            detect_tables: Обнаруживать ли таблицы
            detect_formulas: Обнаруживать ли формулы
        """
        self.config = {
            'extract_images': extract_images,
            'detect_tables': detect_tables,
            'detect_formulas': detect_formulas
        }
        self._parsers = {}  # Кэш созданных парсеров

    def process(self, file_path: str, output_dir: Optional[str] = None) -> Tuple[str, Dict]:
        """
        Обрабатывает документ и возвращает обогащённый текст + метаданные

        Args:
            file_path: Путь к файлу
            output_dir: Директория для сохранения извлечённых элементов

        Returns:
            Tuple[обогащённый_текст, метаданные]
        """
        logger.info(f"Обработка файла: {file_path}")

        # Получаем парсер
        parser = self._get_parser(file_path)
        if not parser:
            raise ValueError(f"Не найден парсер для файла: {file_path}")

        # Парсим документ
        parse_result = parser.parse(file_path, output_dir=output_dir)

        # Форматируем результат
        enriched_text = self._format_for_llm(parse_result)
        metadata = self._extract_metadata(parse_result)

        return enriched_text, metadata

    def _get_parser(self, file_path: str) -> Optional:
        """
        Получает или создаёт парсер для файла
        """
        ext = Path(file_path).suffix.lower()

        if ext not in self._parsers:
            parser = ParserFactory.create_parser(file_path, self.config)
            if parser:
                self._parsers[ext] = parser
                logger.info(f"Создан парсер {parser.parser_name} для {ext}")
            else:
                logger.error(f"Не удалось создать парсер для {ext}")
                return None

        return self._parsers.get(ext)

    def _format_for_llm(self, parse_result: Dict) -> str:
        """
        Форматирует результат парсинга для отправки в LLM
        """
        parts = ['=== ИЗВЛЕЧЁННЫЙ ТЕКСТ ДОКУМЕНТА ===\n', parse_result['text']]

        # Добавляем формулы
        if parse_result['formulas']:
            parts.append('\n\n=== РАСПОЗНАННЫЕ ФОРМУЛЫ ===')
            for i, f in enumerate(parse_result['formulas'], 1):
                parts.append(
                    f'\n[{i}] Стр.{f.page} ({f.formula_type}, conf:{f.confidence:.2f}): ${f.latex}$'
                )
                if f.context:
                    parts.append(f'    Контекст: {f.context}')

        # Добавляем таблицы
        if parse_result['tables']:
            parts.append('\n\n=== ИЗВЛЕЧЁННЫЕ ТАБЛИЦЫ ===')
            for i, t in enumerate(parse_result['tables'], 1):
                parts.append(f'\n[{i}] Стр.{t.page} ({t.num_rows}×{t.num_cols})')
                if t.caption:
                    parts.append(f'    Подпись: {t.caption}')
                parts.append(t.to_markdown())

        # Добавляем изображения
        if parse_result['images']:
            parts.append('\n\n=== ИЗОБРАЖЕНИЯ ===')
            for i, img in enumerate(parse_result['images'], 1):
                desc = f'Стр.{img.page}, {img.width}×{img.height}px, формат: {img.image_format}'
                if img.caption:
                    desc += f', подпись: {img.caption}'
                parts.append(f'[{i}] {desc}')

        return '\n'.join(parts)

    def _extract_metadata(self, parse_result: Dict) -> Dict:
        """
        Извлекает метаданные из результата парсинга
        """
        return {
            'filename': parse_result['metadata']['filename'],
            'pages': parse_result['metadata']['pages'],
            'formulas_count': parse_result['metadata']['formulas_count'],
            'tables_count': parse_result['metadata']['tables_count'],
            'images_count': parse_result['metadata']['images_count'],
            'formulas': [f.to_dict() for f in parse_result['formulas']],
            'tables': [t.to_dict() for t in parse_result['tables']],
            'parser': parse_result['metadata'].get('parser', 'unknown')
        }

    def process_batch(self, file_paths: List[str], output_dir: Optional[str] = None) -> List[Tuple[str, Dict]]:
        """
        Пакетная обработка нескольких файлов
        """
        results = []
        for file_path in file_paths:
            try:
                result = self.process(file_path, output_dir)
                results.append(result)
            except Exception as e:
                logger.error(f"Ошибка при обработке {file_path}: {e}")
                results.append((f"Ошибка: {e}", {'filename': Path(file_path).name, 'error': str(e)}))
        return results

    @property
    def supported_extensions(self) -> List[str]:
        """Список поддерживаемых расширений"""
        return ParserFactory.get_supported_extensions()

    def clear_cache(self) -> None:
        """Очистка кэша парсеров"""
        self._parsers.clear()
        logger.info("Кэш парсеров очищен")
