import os
import logging
from pathlib import Path
from typing import Optional, List, Tuple
import pypdf
from docx import Document

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FileProcessor:
    """Обработчик файлов различных форматов"""

    SUPPORTED_EXTENSIONS = {
        '.pdf': 'PDF документ',
        '.docx': 'Word документ',
        '.doc': 'Word документ (старый формат)',
        '.txt': 'Текстовый файл',
        '.rtf': 'Rich Text Format'
    }

    def __init__(self, max_file_size_mb: int = 10):
        self.max_file_size = max_file_size_mb * 1024 * 1024

    def process(self, file_path: str) -> Tuple[str, dict]:
        """
        Основной метод обработки файла
        Возвращает (текст, метаданные)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл не найден: {file_path}")

        # Проверка размера
        file_size = os.path.getsize(file_path)
        if file_size > self.max_file_size:
            raise ValueError(
                f"Файл слишком большой: {file_size / 1024 / 1024:.1f} МБ (макс. {self.max_file_size / 1024 / 1024:.0f} МБ)")

        # Получение расширения
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Неподдерживаемый формат: {file_ext}. Поддерживаются: {list(self.SUPPORTED_EXTENSIONS.keys())}")

        # Метаданные файла
        metadata = {
            "filename": os.path.basename(file_path),
            "extension": file_ext,
            "size_bytes": file_size,
            "size_mb": round(file_size / 1024 / 1024, 2),
            "created": datetime.fromtimestamp(os.path.getctime(file_path)).isoformat() if hasattr(os.path,
                                                                                                  'getctime') else None,
            "modified": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
        }

        # Извлечение текста в зависимости от формата
        text = self._extract_text(file_path, file_ext)

        # Базовая очистка текста
        text = self._clean_text(text)

        metadata["char_count"] = len(text)
        metadata["word_count"] = len(text.split())

        logger.info(f"Файл {file_path} обработан: {metadata['word_count']} слов")

        return text, metadata

    def _extract_text(self, file_path: str, file_ext: str) -> str:
        """Извлечение текста из файла"""
        try:
            if file_ext == '.pdf':
                return self._extract_from_pdf(file_path)
            elif file_ext in ['.docx', '.doc']:
                return self._extract_from_docx(file_path)
            elif file_ext == '.txt':
                return self._extract_from_txt(file_path)
            elif file_ext == '.rtf':
                return self._extract_from_rtf(file_path)
            else:
                raise ValueError(f"Не реализована обработка формата: {file_ext}")
        except Exception as e:
            logger.error(f"Ошибка при извлечении текста из {file_path}: {str(e)}")
            raise

    def _extract_from_pdf(self, file_path: str) -> str:
        """Извлечение текста из PDF"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    page_text = page.extract_text()
                    if page_text.strip():
                        text += f"--- Страница {page_num} ---\n{page_text}\n\n"
                    else:
                        logger.warning(f"Страница {page_num} не содержит текста (возможно, сканированная)")
            return text
        except Exception as e:
            logger.error(f"Ошибка чтения PDF: {str(e)}")
            raise

    def _extract_from_docx(self, file_path: str) -> str:
        """Извлечение текста из DOCX"""
        try:
            doc = Document(file_path)
            text = []

            # Извлечение текста из параграфов
            for para in doc.paragraphs:
                if para.text.strip():
                    text.append(para.text)

            # Извлечение текста из таблиц
            for table in doc.tables:
                for row in table.rows:
                    row_text = [cell.text for cell in row.cells if cell.text.strip()]
                    if row_text:
                        text.append(" | ".join(row_text))

            return "\n".join(text)
        except Exception as e:
            logger.error(f"Ошибка чтения DOCX: {str(e)}")
            raise

    def _extract_from_txt(self, file_path: str) -> str:
        """Извлечение текста из TXT"""
        encodings = ['utf-8', 'cp1251', 'koi8-r', 'latin-1']

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    return file.read()
            except UnicodeDecodeError:
                continue

        raise ValueError(f"Не удалось прочитать файл {file_path} ни в одной из кодировок: {encodings}")

    def _extract_from_rtf(self, file_path: str) -> str:
        """Простое извлечение текста из RTF (удаляет управляющие символы)"""
        try:
            with open(file_path, 'r', encoding='cp1251', errors='ignore') as file:
                content = file.read()
                # Очень простая очистка RTF
                import re
                # Удаление RTF команд
                text = re.sub(r'\\[a-z]+[\d]*', ' ', content)
                text = re.sub(r'[{}]', '', text)
                return text
        except Exception as e:
            logger.error(f"Ошибка чтения RTF: {str(e)}")
            raise

    def _clean_text(self, text: str) -> str:
        """Очистка и нормализация текста"""
        if not text:
            return ""

        # Замена множественных пробелов
        import re
        text = re.sub(r'\s+', ' ', text)

        # Удаление странных символов
        text = re.sub(r'[^\w\s.,!?;:""''()\[\]{}«»—–-]', '', text)

        return text.strip()


from datetime import datetime