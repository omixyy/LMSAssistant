import fitz
import re
import logging
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

from .base_parser import Parser
from .non_text_elements import Formula, Table, Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import pdfplumber

    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logger.warning('pdfplumber не установлен. Таблицы будут извлекаться только через PyMuPDF.')


class PDFParser(Parser):
    """
    Парсер для PDF документов с извлечением формул, таблиц и изображений
    """

    @property
    def supported_extensions(self) -> List[str]:
        return ['.pdf']

    @property
    def parser_name(self) -> str:
        return 'PDFParser'

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self._init_formula_patterns()
        self._init_math_symbols()

        # Для pdfplumber нужно открывать файл отдельно
        self._pdfplumber_doc = None

    def _init_formula_patterns(self) -> None:
        """Инициализация паттернов для поиска формул"""
        self.formula_patterns = [
            (r'\\\[([\s\S]*?)\\\]', 'display'),  # \[ ... \]
            (r'\\\(([\s\S]*?)\\\)', 'inline'),  # \( ... \)
            (r'\$\$([\s\S]*?)\$\$', 'display'),  # $$ ... $$
            (r'(?<!\$)\$([^\$]+)\$(?!\$)', 'inline'),  # $ ... $ (не вложенные)
            (r'\\begin\{(equation|align|gather|multline)\*?\}([\s\S]*?)\\end\{\1\}', 'display'),
        ]

    def _init_math_symbols(self) -> None:
        """Инициализация математических символов и команд"""
        self.math_symbols = set('=+-*/^_{}[]()⟨⟩∑∏∫∂∇∞≈≠≤≥±×÷√')
        self.latex_commands = {
            'frac', 'sqrt', 'sum', 'prod', 'int', 'oint', 'partial', 'nabla',
            'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'theta', 'lambda',
            'mu', 'pi', 'sigma', 'omega', 'infty', 'cdot', 'times', 'leq', 'geq',
            'neq', 'approx', 'equiv', 'subset', 'supset', 'cup', 'cap', 'forall',
            'exists', 'nabla', 'langle', 'rangle', 'left', 'right', 'begin', 'end'
        }

    def parse(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """
        Парсинг PDF документа

        Args:
            file_path: Путь к PDF файлу
            **kwargs:
                - output_dir: директория для сохранения изображений
                - extract_text_only: только текст (без анализа)
        """
        self._log_parse_start(file_path)

        output_dir = kwargs.get('output_dir')
        extract_text_only = kwargs.get('extract_text_only', False)

        # Открываем PDF через fitz
        doc = fitz.open(file_path)

        # Для pdfplumber открываем отдельно, если нужно извлекать таблицы
        pdfplumber_doc = None
        if self.detect_tables and PDFPLUMBER_AVAILABLE and not extract_text_only:
            try:
                pdfplumber_doc = pdfplumber.open(file_path)
                logger.debug('pdfplumber успешно открыл документ')
            except Exception as e:
                logger.warning(f'Не удалось открыть PDF через pdfplumber: {e}')
                pdfplumber_doc = None

        result = self._init_result_dict(Path(file_path).name)
        result['metadata']['pages'] = len(doc)

        all_text = []

        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                logger.debug(f'Обработка страницы {page_num + 1}/{len(doc)}')

                if extract_text_only:
                    # Только текст, без анализа
                    text = page.get_text()
                    all_text.append(f'\n--- Страница {page_num + 1} ---\n{text}')
                else:
                    # Полный анализ
                    text_dict = page.get_text('dict')

                    # Получаем соответствующую страницу из pdfplumber для таблиц
                    plumb_page = None
                    if pdfplumber_doc and page_num < len(pdfplumber_doc.pages):
                        plumb_page = pdfplumber_doc.pages[page_num]

                    # Извлечение элементов
                    page_elements = self._extract_page_elements(
                        page, plumb_page, text_dict, page_num + 1, output_dir
                    )

                    # Добавление результатов
                    result['formulas'].extend(page_elements['formulas'])
                    result['tables'].extend(page_elements['tables'])
                    result['images'].extend(page_elements['images'])

                    # Сборка текста страницы
                    page_text = self._build_page_text(
                        text_dict,
                        page_elements['formulas'],
                        page_elements['table_bboxes']
                    )

                    if page_text.strip():
                        all_text.append(f'\n--- Страница {page_num + 1} ---\n{page_text}')
        finally:
            # Закрываем документы
            doc.close()
            if pdfplumber_doc:
                pdfplumber_doc.close()

        result['text'] = '\n'.join(all_text)
        result['metadata']['formulas_count'] = len(result['formulas'])
        result['metadata']['tables_count'] = len(result['tables'])
        result['metadata']['images_count'] = len(result['images'])

        self._log_parse_complete(result)
        return result

    def _extract_page_elements(self, fitz_page, plumb_page, text_dict: Dict,
                               page_num: int, output_dir: Optional[str]) -> Dict:
        """Извлечение всех элементов со страницы"""
        elements = {
            'formulas': [],
            'tables': [],
            'images': [],
            'table_bboxes': []
        }

        # Извлечение формул
        if self.detect_formulas:
            elements['formulas'] = self._extract_formulas(fitz_page, text_dict, page_num)

        # Извлечение таблиц
        if self.detect_tables:
            # Сначала пробуем через PyMuPDF
            tables = self._extract_tables_with_fitz(fitz_page, page_num)

            # Если не нашли, пробуем через pdfplumber
            if not tables and plumb_page:
                tables = self._extract_tables_with_pdfplumber(plumb_page, page_num)

            elements['tables'] = tables
            elements['table_bboxes'] = [t.bbox for t in tables]

        # Извлечение изображений
        if self.extract_images:
            elements['images'] = self._extract_images(fitz_page, page_num, output_dir)

        return elements

    def _extract_formulas(self, page, text_dict: Dict, page_num: int) -> List[Formula]:
        """Извлечение формул со страницы"""
        formulas = []
        formula_bboxes = []

        # Поиск формул в текстовых блоках
        for block in text_dict.get('blocks', []):
            if block.get('type') != 0:  # Только текстовые блоки
                continue

            for line in block.get('lines', []):
                for span in line.get('spans', []):
                    text = span.get('text', '').strip()
                    if not text or len(text) < 2:
                        continue

                    formula_info = self._detect_formula(text, span)
                    if formula_info['is_formula']:
                        formula = Formula(
                            page=page_num,
                            text=text,
                            latex=formula_info['latex'],
                            bbox=tuple(span['bbox']),
                            context=self._get_context(text_dict, span),
                            confidence=formula_info['confidence'],
                            formula_type=formula_info['type']
                        )
                        formulas.append(formula)
                        formula_bboxes.append(tuple(span['bbox']))

        # Поиск центрированных формул (display formulas)
        formulas.extend(self._find_centered_formulas(page, text_dict, page_num, formula_bboxes))

        return formulas

    def _extract_tables_with_fitz(self, page, page_num: int) -> List[Table]:
        """Извлечение таблиц через PyMuPDF"""
        tables = []

        try:
            # Проверяем наличие метода find_tables
            if hasattr(page, 'find_tables'):
                pymupdf_tables = page.find_tables()
                for tab in pymupdf_tables:
                    if tab.bbox and hasattr(tab, 'extract') and tab.extract():
                        data = tab.extract()
                        # Фильтрация пустых таблиц
                        if data and any(
                                any(
                                    cell and str(cell).strip()
                                    for cell in row
                                ) for row in data
                        ):
                            table = Table(
                                page=page_num,
                                bbox=tuple(tab.bbox),
                                data=data,
                                caption=self._find_caption_near_bbox(page, tab.bbox)
                            )
                            tables.append(table)

        except AttributeError as e:
            logger.warning(f'find_tables не доступен в этой версии PyMuPDF: {e}')

        except Exception as e:
            logger.warning(f'Ошибка при извлечении таблиц через PyMuPDF: {e}')

        return tables

    def _extract_tables_with_pdfplumber(self, page, page_num: int) -> List[Table]:
        """Извлечение таблиц через pdfplumber"""
        tables = []

        try:
            # Поиск таблиц на странице
            pdfplumber_tables = page.find_tables()

            for pt in pdfplumber_tables:
                data = pt.extract()
                if data and any(any(c and str(c).strip() for c in row) for row in data):
                    bbox = (pt.bbox[0], pt.bbox[1], pt.bbox[2], pt.bbox[3])
                    table = Table(
                        page=page_num,
                        bbox=bbox,
                        data=data,
                        caption=self._find_caption_near_bbox_with_coords(page, bbox)
                    )
                    tables.append(table)
        except Exception as e:
            logger.warning(f'pdfplumber не смог извлечь таблицы: {e}')

        return tables

    def _extract_images(self, page, page_num: int, output_dir: Optional[str]) -> List[Image]:
        """Извлечение изображений со страницы"""
        images = []
        image_list = page.get_images(full=True)

        for img_idx, img in enumerate(image_list):
            xref = img[0]

            try:
                img_info = page.parent.extract_image(xref)
                if not img_info:
                    continue

                image_bytes = img_info.get('image')
                image_ext = img_info.get('ext', 'png')
                rects = page.get_image_rects(xref)

                if not rects:
                    continue

                for rect_idx, rect in enumerate(rects):
                    bbox = (rect.x0, rect.y0, rect.x1, rect.y1)
                    pix = fitz.Pixmap(page.parent, xref)

                    extracted_img = Image(
                        page=page_num,
                        bbox=bbox,
                        image_data=image_bytes if len(image_bytes) < 10_000_000 else None,
                        image_format=image_ext,
                        caption=self._find_caption_near_bbox(page, bbox),
                        xref=xref,
                        width=pix.width,
                        height=pix.height
                    )
                    images.append(extracted_img)

                    # Сохранение на диск
                    if output_dir and image_bytes:
                        self._save_image(output_dir, page_num, img_idx, rect_idx, image_ext, image_bytes)

                    pix = None

            except Exception as e:
                logger.error(f'Ошибка при обработке изображения {xref}: {e}')
                continue

        return images

    def _find_centered_formulas(self, page, text_dict: Dict, page_num: int, existing_bboxes: List) -> List[Formula]:
        """Поиск центрированных формул"""
        formulas = []
        page_width = page.rect.width

        for block in text_dict.get('blocks', []):
            if block.get('type') != 0:
                continue

            bbox = block.get('bbox')
            if not bbox:
                continue

            # Проверка на центрирование
            block_center = (bbox[0] + bbox[2]) / 2
            is_centered = abs(block_center - page_width / 2) < page_width * 0.1

            block_text = ' '.join(
                span.get('text', '')
                for line in block.get('lines', [])
                for span in line.get('spans', [])
            ).strip()

            # Отсеиваем очевидно текстовые блоки (сплошная проза без математики)
            if self._looks_like_prose(block_text):
                continue

            # Отсеиваем нумерованные списки без явной математики (например "1. Постройте гистограмму...")
            if re.match(r'^\s*\d+[\.\)]\s+', block_text) and not self._has_math_operators(block_text):
                continue

            if (is_centered and
                    len(block_text) < 200 and
                    self._has_math_content(block_text) and
                    not any(self._bbox_overlaps(tuple(bbox), eb) for eb in existing_bboxes)):
                formulas.append(Formula(
                    page=page_num,
                    text=block_text,
                    latex=self._text_to_latex(block_text),
                    bbox=tuple(bbox),
                    context='',
                    confidence=0.7,
                    formula_type='display'
                ))

        return formulas

    def _detect_formula(self, text: str, span: Dict) -> Dict:
        """Определяет, является ли текст формулой"""
        result = {
            'is_formula': False,
            'latex': text,
            'type': 'inline',
            'confidence': 0.0
        }

        text_stripped = text.strip()

        # Проверка на явные LaTeX-маркеры
        for pattern, ftype in self.formula_patterns:
            match = re.search(pattern, text_stripped, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(1) if match.lastindex else match.group(0)
                if self._has_math_operators(content):
                    result.update({
                        'is_formula': True,
                        'type': ftype,
                        'confidence': 0.95,
                        'latex': content.strip()
                    })
                    return result

        # Отсеиваем очевидно обычный текст (длинные слова, знаки препинания, служебные слова)
        if self._looks_like_prose(text_stripped):
            return result

        # Если нет ни математических операторов, ни «сильных» математических символов (кроме скобок),
        # то с большой вероятностью это не формула
        if (not self._has_math_operators(text_stripped) and
                not any(ch in (self.math_symbols - set('()')) for ch in text_stripped)):
            return result

        # Дополнительный фильтр: нумерованные пункты без математики (например "3. Вычислить среднее значение...")
        if re.match(r'^\s*\d+[\.\)]\s+', text_stripped) and not self._has_math_operators(text_stripped):
            return result

        # Вычисляем вероятность
        score = self._calculate_math_score(text_stripped, span)

        if score >= 0.6:
            result.update({
                'is_formula': True,
                'type': 'display' if score >= 0.8 else 'inline',
                'confidence': min(score, 1.0),
                'latex': self._text_to_latex(text_stripped)
            })

        return result

    def _calculate_math_score(self, text: str, span: Dict) -> float:
        """Вычисляет вероятность того, что текст является формулой"""
        score = 0.0

        # LaTeX-команды
        latex_cmds = re.findall(r'\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^}]*\})?', text)
        if latex_cmds:
            score += 0.5 + min(len(latex_cmds) * 0.15, 0.4)

        # Математические операторы
        if self._has_math_operators(text):
            score += 0.3

        # Структурные признаки
        if re.search(r'[a-zA-Z]\s*[_^]\s*[\{a-zA-Z0-9]', text):
            score += 0.25
        if re.search(r'\{[^}]+\}[\s]*\{[^}]+\}', text):
            score += 0.2

        # Шрифт
        font = span.get('font', '').lower()
        if any(kw in font for kw in ['math', 'symbol', 'cmsy', 'msam']):
            score += 0.15

        return score

    def _build_page_text(self, text_dict: Dict, formulas: List[Formula], table_bboxes: List[Tuple]) -> str:
        """Сборка текста страницы с маркерами"""
        lines = []
        exclude_bboxes = [f.bbox for f in formulas] + table_bboxes

        for block in text_dict.get('blocks', []):
            if block.get('type') != 0:
                continue

            for line in block.get('lines', []):
                line_parts = []
                for span in line.get('spans', []):
                    text = span.get('text', '').strip()
                    if not text:
                        continue

                    if self._bbox_overlaps_any(tuple(span['bbox']), exclude_bboxes):
                        continue

                    line_parts.append(text)

                if line_parts:
                    lines.append(' '.join(line_parts))

        # Добавляем маркеры для формул
        for f in formulas:
            lines.append(f'«ФОРМУЛА {f.formula_type}: ${f.latex}$»')

        return '\n'.join(lines)

    # Вспомогательные статические методы
    @staticmethod
    def _has_math_operators(text: str) -> bool:
        patterns = [
            r'[=≠<>]\s*[a-zA-Z\\]',
            r'[a-zA-Z]\s*[=≠<>]',
            r'[\^_]\s*\{',
            r'\\[a-z]+\{',
            r'\d+\s*[/÷×]\s*\d+',
            r'[∑∏∫∂][\s_]*[\{a-zA-Z]',
        ]
        return any(re.search(p, text) for p in patterns)

    @staticmethod
    def _looks_like_prose(text: str) -> bool:
        if len(text) < 3:
            return False
        if re.search(r'[а-яa-z]{10,}', text, re.IGNORECASE):
            return True
        if re.search(r'[.,;:]|\b(и|или|на|в|для|пример|исследование)\b', text, re.IGNORECASE):
            return True
        if text.count(' ') > len(text) * 0.3:
            return True
        return False

    @staticmethod
    def _text_to_latex(text: str) -> str:
        latex = text
        replacements = {
            '≠': r'\neq', '≤': r'\leq', '≥': r'\geq', '±': r'\pm',
            '×': r'\times', '÷': r'\div', '√': r'\sqrt{}',
            '∑': r'\sum', '∏': r'\prod', '∫': r'\int', '∂': r'\partial',
            '∇': r'\nabla', '∞': r'\infty', '≈': r'\approx',
        }
        for orig, latex_eq in replacements.items():
            latex = latex.replace(orig, latex_eq)
        return latex

    def _has_math_content(self, text: str) -> bool:
        if any(c in text for c in self.math_symbols):
            return True
        if re.search(r'\\[a-z]+', text, re.IGNORECASE):
            return True
        text_lower = text.lower()
        return any(kw in text_lower for kw in self.latex_commands)

    @staticmethod
    def _find_caption_near_bbox(page, bbox: Tuple, max_distance: float = 50) -> str:
        """Поиск подписи рядом с bbox"""
        x0, y0, x1, y1 = bbox
        text = page.get_text('dict')

        caption_patterns = [
            r'(?:Рисунок|Рис\.|Рис)\s*\.?\s*\d+\s*[:.—-]?\s*([^\n\.]+)',
            r'(?:Figure|Fig\.)\s*\.?\s*\d+\s*[:.—-]?\s*([^\n\.]+)',
            r'(?:Таблица|Табл\.|Таблиц)\s*\.?\s*\d+\s*[:.—-]?\s*([^\n\.]+)',
            r'(?:Table|Tab\.)\s*\.?\s*\d+\s*[:.—-]?\s*([^\n\.]+)',
        ]

        candidates = []

        for block in text.get('blocks', []):
            if block.get('type') != 0:
                continue

            block_bbox = block.get('bbox')
            if not block_bbox:
                continue

            block_y = (block_bbox[1] + block_bbox[3]) / 2
            elem_y = (y0 + y1) / 2

            if abs(block_y - elem_y) > max_distance:
                continue

            block_text = ' '.join(
                span.get('text', '').strip()
                for line in block.get('lines', [])
                for span in line.get('spans', [])
            ).strip()

            if not block_text or len(block_text) > 300:
                continue

            for pattern in caption_patterns:
                match = re.search(pattern, block_text, re.IGNORECASE | re.MULTILINE)
                if match:
                    caption = match.group(1).strip() if match.lastindex else block_text
                    candidates.append((abs(block_y - elem_y), caption))
                    break

        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1]

        return ''

    @staticmethod
    def _find_caption_near_bbox_with_coords(page, bbox: Tuple, max_distance: float = 50) -> str:
        """
        Поиск подписи рядом с bbox для pdfplumber страницы
        """
        # Упрощённая версия для pdfplumber
        try:
            words = page.extract_words()
            caption_text = []
            bbox_center_y = (bbox[1] + bbox[3]) / 2

            for word in words:
                word_center_y = (word['top'] + word['bottom']) / 2
                if abs(word_center_y - bbox_center_y) < max_distance:
                    caption_text.append(word['text'])

            if caption_text:
                return ' '.join(caption_text)
        except:
            pass

        return ''

    @staticmethod
    def _get_context(text_dict: Dict, span: Dict, max_words: int = 10) -> str:
        """Получение контекста вокруг span"""
        context_words = []
        target_bbox = span['bbox']

        for block in text_dict.get('blocks', []):
            if block.get('type') != 0:
                continue
            for line in block.get('lines', []):
                for s in line.get('spans', []):
                    s_bbox = s['bbox']
                    if abs(s_bbox[1] - target_bbox[1]) < 5:
                        words = s.get('text', '').split()
                        context_words.extend(words)

        return ' '.join(context_words[:max_words])

    @staticmethod
    def _save_image(output_dir: str, page_num: int, img_idx: int, rect_idx: int, ext: str, data: bytes) -> None:
        """Сохранение изображения на диск"""
        img_path = Path(output_dir) / f'page_{page_num}_img_{img_idx}_{rect_idx}.{ext}'
        with open(img_path, 'wb') as f:
            f.write(data)
        logger.debug(f'Изображение сохранено: {img_path}')

    @staticmethod
    def _bbox_overlaps_any(bbox: Tuple, bbox_list: List[Tuple], threshold: float = 0.5) -> bool:
        """Проверка перекрытия bbox с любым из списка"""
        if not bbox or not bbox_list:
            return False

        x0, y0, x1, y1 = bbox

        for other in bbox_list:
            ox0, oy0, ox1, oy1 = other
            if not (x1 < ox0 or x0 > ox1 or y1 < oy0 or y0 > oy1):
                inter_x0 = max(x0, ox0)
                inter_y0 = max(y0, oy0)
                inter_x1 = min(x1, ox1)
                inter_y1 = min(y1, oy1)

                if inter_x0 < inter_x1 and inter_y0 < inter_y1:
                    inter_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
                    self_area = (x1 - x0) * (y1 - y0)
                    if self_area > 0 and inter_area / self_area >= threshold:
                        return True
        return False

    @staticmethod
    def _bbox_overlaps(bbox1: Tuple, bbox2: Tuple, threshold: float = 0.5) -> bool:
        """Проверка перекрытия двух bbox"""
        return PDFParser._bbox_overlaps_any(bbox1, [bbox2], threshold)
