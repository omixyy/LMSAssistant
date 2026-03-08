import fitz  # PyMuPDF
import re
import logging
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import base64
import io

from .non_text_elements import Formula, Table, ExtractedImage

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False


class AdvancedPDFParser:
    """
    Продвинутый парсер PDF с извлечением формул, таблиц и изображений
    """
    
    def __init__(self, extract_images: bool = True, detect_tables: bool = True):
        self.extract_images = extract_images
        self.detect_tables = detect_tables
        
        # Паттерны для поиска LaTeX-подобных формул в тексте
        self.formula_patterns = [
            (r'\\\[([\s\S]*?)\\\]', 'display'),      # \[ ... \]
            (r'\\\(([\s\S]*?)\\\)', 'inline'),        # \( ... \)
            (r'\$\$([\s\S]*?)\$\$', 'display'),       # $$ ... $$
            (r'(?<!\$)\$([^\$]+)\$(?!\$)', 'inline'), # $ ... $ (не вложенные)
            (r'\\begin\{(equation|align|gather|multline)\*?\}([\s\S]*?)\\end\{\1\}', 'display'),
        ]
        
        # Математические символы и команды LaTeX
        self.math_symbols = set('=+-*/^_{}[]()⟨⟩∑∏∫∂∇∞≈≠≤≥±×÷√')
        self.latex_commands = {
            'frac', 'sqrt', 'sum', 'prod', 'int', 'oint', 'partial', 'nabla',
            'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'theta', 'lambda',
            'mu', 'pi', 'sigma', 'omega', 'infty', 'cdot', 'times', 'leq', 'geq',
            'neq', 'approx', 'equiv', 'subset', 'supset', 'cup', 'cap', 'forall',
            'exists', 'nabla', 'langle', 'rangle', 'left', 'right', 'begin', 'end'
        }
        
        # Паттерны для детекции математических выражений
        self.math_heuristics = [
            r'[a-zA-Z]\s*[=≠<>]\s*[a-zA-Z0-9]',  # x = y
            r'\\[a-z]+\{',  # LaTeX команды с аргументами
            r'\^[{0-9a-zA-Z}]',  # Степени
            r'_{[{0-9a-zA-Z},]+}',  # Индексы
            r'\d+\s*[/÷]\s*\d+',  # Дроби
            r'[a-zA-Z]+\s*\([^)]*\)',  # Функции типа f(x)
        ]
    
    def parse_pdf(self, pdf_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Основной метод парсинга PDF
        
        Args:
            pdf_path: Путь к PDF файлу
            output_dir: Директория для сохранения извлечённых изображений (опционально)
        """
        logger.info(f"Начинаю парсинг PDF: {pdf_path}")
        
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        doc = fitz.open(pdf_path)
        
        result = {
            'text': '',
            'formulas': [],
            'tables': [],
            'images': [],
            'metadata': {
                'pages': len(doc),
                'formulas_count': 0,
                'tables_count': 0,
                'images_count': 0,
                'filename': Path(pdf_path).name
            }
        }
        
        all_text = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            logger.info(f"Обработка страницы {page_num + 1}/{len(doc)}")
            
            # === 1. Извлечение текста с позиционированием ===
            text_dict = page.get_text("dict")
            page_text_parts = []
            
            # === 2. Детекция и извлечение формул ===
            formulas_on_page = self._extract_formulas(page, text_dict)
            formula_bboxes = [f.bbox for f in formulas_on_page]
            result['formulas'].extend(formulas_on_page)
            result['metadata']['formulas_count'] += len(formulas_on_page)
            
            # === 3. Извлечение таблиц ===
            if self.detect_tables:
                tables_on_page = self._extract_tables(page, page_num + 1)
                table_bboxes = [t.bbox for t in tables_on_page]
                result['tables'].extend(tables_on_page)
                result['metadata']['tables_count'] += len(tables_on_page)
            else:
                table_bboxes = []
            
            # === 4. Извлечение изображений ===
            if self.extract_images:
                images_on_page = self._extract_images(page, page_num + 1, output_dir)
                result['images'].extend(images_on_page)
                result['metadata']['images_count'] += len(images_on_page)
            
            # === 5. Сборка текста страницы с маркерами ===
            page_text = self._build_page_text(
                text_dict, 
                formulas_on_page, 
                table_bboxes,
                exclude_bboxes=formula_bboxes + table_bboxes
            )
            
            if page_text.strip():
                all_text.append(f"\n--- Страница {page_num + 1} ---\n{page_text}")
        
        doc.close()
        
        result['text'] = '\n'.join(all_text)
        logger.info(f"Парсинг завершён. Формул: {result['metadata']['formulas_count']}, "
                   f"Таблиц: {result['metadata']['tables_count']}, "
                   f"Изображений: {result['metadata']['images_count']}")
        
        return result
    
    def _extract_formulas(self, page, text_dict: Dict) -> List[Formula]:
        """Извлечение формул со страницы"""
        formulas = []
        
        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:  # Только текстовые блоки
                continue
                
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text or len(text) < 2:
                        continue
                    
                    formula_info = self._detect_formula(text, span)
                    if formula_info['is_formula']:
                        formula = Formula(
                            page=page.number + 1,
                            text=text,
                            latex=formula_info['latex'],
                            bbox=tuple(span["bbox"]),
                            context=self._get_context(text_dict, span),
                            confidence=formula_info['confidence'],
                            formula_type=formula_info['type']
                        )
                        formulas.append(formula)
        
        # Дополнительный поиск: блоки с центрированным текстом и матем. символами
        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            bbox = block.get("bbox")
            if not bbox:
                continue
                
            # Проверка на центрированный блок (возможная display-формула)
            page_width = page.rect.width
            block_center = (bbox[0] + bbox[2]) / 2
            is_centered = abs(block_center - page_width/2) < page_width * 0.1
            
            block_text = " ".join(
                span.get("text", "") 
                for line in block.get("lines", []) 
                for span in line.get("spans", [])
            ).strip()
            
            if (is_centered and 
                len(block_text) < 200 and 
                self._has_math_content(block_text) and
                not any(f.bbox == tuple(bbox) for f in formulas)):
                
                formulas.append(Formula(
                    page=page.number + 1,
                    text=block_text,
                    latex=self._text_to_latex(block_text),
                    bbox=tuple(bbox),
                    context="",
                    confidence=0.7,
                    formula_type="display"
                ))
        
        return formulas
    
    def _detect_formula(self, text: str, span: Dict) -> Dict:
        """Определяет, является ли текст формулой — с защитой от ложных срабатываний"""
        result = {
            'is_formula': False,
            'latex': text,
            'type': 'unknown',
            'confidence': 0.0
        }
        
        text_stripped = text.strip()
        
        # === БЛОК 1: Явные LaTeX-маркеры (высокая уверенность) ===
        for pattern, ftype in self.formula_patterns:
            match = re.search(pattern, text_stripped, re.DOTALL | re.IGNORECASE)
            if match:
                # Дополнительная проверка: внутри должны быть реальные математические символы
                content = match.group(1) if match.lastindex else match.group(0)
                if self._has_math_operators(content):  # новая функция, см. ниже
                    result.update({
                        'is_formula': True,
                        'type': ftype,
                        'confidence': 0.95,
                        'latex': content.strip()
                    })
                    return result
        
        # === БЛОК 2: Строгая эвристика для "подозрительного" текста ===
        # Отклоняем сразу, если текст похож на обычное предложение
        if self._looks_like_prose(text_stripped):
            return result  # is_formula=False
        
        # Считаем "математичность" текста
        score = 0.0
        
        # 1. LaTeX-команды (сильный сигнал)
        latex_cmds = re.findall(r'\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^}]*\})?', text_stripped)
        if latex_cmds:
            score += 0.5 + min(len(latex_cmds) * 0.15, 0.4)
        
        # 2. Математические операторы (не просто символы, а в контексте)
        if self._has_math_operators(text_stripped):
            score += 0.3
        
        # 3. Структурные признаки формул
        if re.search(r'[a-zA-Z]\s*[_^]\s*[\{a-zA-Z0-9]', text_stripped):  # индексы/степени
            score += 0.25
        if re.search(r'\{[^}]+\}[\s]*\{[^}]+\}', text_stripped):  # вложенные группы
            score += 0.2
        
        # 4. Шрифт (дополнительный сигнал, но не решающий)
        font = span.get("font", "").lower()
        if any(kw in font for kw in ['math', 'symbol', 'cmsy', 'msam']):  # специфичные для формул
            score += 0.15
        
        # === Порог срабатывания + защита от ложных срабатываний ===
        if score >= 0.6 and not self._contains_only_words(text_stripped):
            result.update({
                'is_formula': True,
                'type': 'display' if score >= 0.8 else 'inline',
                'confidence': min(score, 1.0),
                'latex': self._text_to_latex(text_stripped)
            })
        
        return result


    def _has_math_operators(self, text: str) -> bool:
        """Проверяет наличие реальных математических операторов, а не просто символов"""
        # Игнорируем одиночные символы = + - в обычном тексте
        # Требуем комбинации, характерные для формул
        patterns = [
            r'[=≠<>]\s*[a-zA-Z\\]',  # = x, \alpha
            r'[a-zA-Z]\s*[=≠<>]',    # x =
            r'[\^_]\s*\{',           # ^{, _{
            r'\\[a-z]+\{',           # \frac{, \sqrt{
            r'\d+\s*[/÷×]\s*\d+',    # 1/2, 3×4
            r'[∑∏∫∂][\s_]*[\{a-zA-Z]', # интегралы, суммы с пределами
        ]
        return any(re.search(p, text) for p in patterns)

    def _looks_like_prose(self, text: str) -> bool:
        """Отсекает обычный текст: предложения с точками, запятыми, длинными словами"""
        if len(text) < 3:
            return False
        
        # Слишком много букв подряд → скорее всего текст
        if re.search(r'[а-яa-z]{10,}', text, re.IGNORECASE):
            return True
        
        # Есть знаки препинания, характерные для предложений
        if re.search(r'[.,;:]|\b(и|или|на|в|для|пример|исследование)\b', text, re.IGNORECASE):
            return True
        
        # Много пробелов между словами → обычный текст
        if text.count(' ') > len(text) * 0.3:
            return True
        
        return False

    def _contains_only_words(self, text: str) -> bool:
        """Проверяет, состоит ли текст ТОЛЬКО из слов без математических операторов"""
        # Удаляем все математические символы и команды
        cleaned = re.sub(r'[=+\-*/^_{}\[\]\\$∑∏∫∂∇∞≈≠≤≥±×÷√]|\\[a-z]+', '', text, flags=re.IGNORECASE)
        cleaned = cleaned.strip()
        
        # Если после очистки осталось только буквы/пробелы/дефисы → это текст
        return bool(re.match(r'^[\sа-яa-zA-Z\-–—]+$', cleaned, re.IGNORECASE))

    def _text_to_latex(self, text: str) -> str:
        """Эвристика для конвертации текста в LaTeX (упрощённая)"""
        # Это базовая реализация; для продакшена используйте MathPix/Nougat
        latex = text
        
        # Простые замены
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
        """Проверяет наличие математического контента"""
        if any(c in text for c in self.math_symbols):
            return True
        if re.search(r'\\[a-z]+', text, re.IGNORECASE):
            return True
        text_lower = text.lower()
        return any(kw in text_lower for kw in self.latex_commands)
    
    def _extract_tables(self, page, page_num: int) -> List[Table]:
        """Извлечение таблиц со страницы"""
        tables = []
        
        # Метод 1: PyMuPDF find_tables (версия 1.20.0+)
        try:
            pymupdf_tables = page.find_tables()
            for tab in pymupdf_tables:
                if tab.bbox and tab.extract():
                    data = tab.extract()
                    # Фильтрация пустых таблиц
                    if any(any(cell and cell.strip() for cell in row) for row in data):
                        table = Table(
                            page=page_num,
                            bbox=tuple(tab.bbox),
                            data=data,
                            num_rows=len(data),
                            num_cols=len(data[0]) if data else 0,
                            caption=self._find_caption_near_bbox(page, tab.bbox)
                        )
                        tables.append(table)
        except AttributeError:
            logger.warning("find_tables не доступен в этой версии PyMuPDF")
        
        # Метод 2: Резервный вариант с pdfplumber
        if not tables and PDFPLUMBER_AVAILABLE:
            try:
                import pdfplumber
                with pdfplumber.open(page.parent) as pdf:
                    plumb_page = pdf.pages[page.number]
                    plumb_tables = plumb_page.find_tables()
                    
                    for pt in plumb_tables:
                        data = pt.extract()
                        if data and any(any(c and c.strip() for c in row) for row in data):
                            bbox = (pt.bbox[0], pt.bbox[1], pt.bbox[2], pt.bbox[3])
                            table = Table(
                                page=page_num,
                                bbox=bbox,
                                data=data,
                                num_rows=len(data),
                                num_cols=len(data[0]) if data else 0,
                                caption=self._find_caption_near_bbox(page, bbox)
                            )
                            tables.append(table)
            except Exception as e:
                logger.warning(f"pdfplumber не смог извлечь таблицы: {e}")
        
        return tables
    
    def _extract_images(self, page, page_num: int, output_dir: Optional[str]) -> List[ExtractedImage]:
        """Извлечение изображений со страницы"""
        images = []
        page_rect = page.rect
        
        # Извлечение растровых изображений
        image_list = page.get_images(full=True)
        
        for img_idx, img in enumerate(image_list):
            xref = img[0]
            
            try:
                # Получаем информацию об изображении
                img_info = page.parent.extract_image(xref)
                if not img_info:
                    continue
                
                image_bytes = img_info.get("image")
                image_ext = img_info.get("ext", "png")
                
                # Находим все вхождения изображения на странице
                rects = page.get_image_rects(xref)
                if not rects:
                    continue
                
                for rect_idx, rect in enumerate(rects):
                    # Конвертируем rect в bbox
                    bbox = (rect.x0, rect.y0, rect.x1, rect.y1)
                    
                    # Получаем размеры
                    pix = fitz.Pixmap(page.parent, xref)
                    width, height = pix.width, pix.height
                    
                    # Поиск подписи
                    caption = self._find_caption_near_bbox(page, bbox)
                    
                    extracted_img = ExtractedImage(
                        page=page_num,
                        bbox=bbox,
                        image_data=image_bytes if len(image_bytes) < 10_000_000 else None,  # Не сохраняем слишком большие
                        image_format=image_ext,
                        caption=caption,
                        xref=xref,
                        width=width,
                        height=height
                    )
                    images.append(extracted_img)
                    
                    # Сохранение на диск если указана директория
                    if output_dir and image_bytes:
                        img_path = Path(output_dir) / f"page_{page_num}_img_{img_idx}_{rect_idx}.{image_ext}"
                        with open(img_path, "wb") as f:
                            f.write(image_bytes)
                        logger.debug(f"Изображение сохранено: {img_path}")
                    
                    pix = None  # Освобождаем память
                    
            except Exception as e:
                logger.error(f"Ошибка при обработке изображения {xref}: {e}")
                continue
        
        # Дополнительно: поиск векторных графиков/диаграмм по признакам
        # (можно расширить анализом paths на странице)
        
        return images
    
    def _find_caption_near_bbox(self, page, bbox: Tuple, max_distance: float = 50) -> str:
        """Поиск подписи рядом с заданным bbox"""
        x0, y0, x1, y1 = bbox
        text = page.get_text("dict")
        
        # Паттерны для подписей
        caption_patterns = [
            r'(?:Рисунок|Рис\.|Рис)\s*\.?\s*\d+\s*[:.—-]?\s*([^\n\.]+)',
            r'(?:Figure|Fig\.)\s*\.?\s*\d+\s*[:.—-]?\s*([^\n\.]+)',
            r'(?:Таблица|Табл\.|Таблиц)\s*\.?\s*\d+\s*[:.—-]?\s*([^\n\.]+)',
            r'(?:Table|Tab\.)\s*\.?\s*\d+\s*[:.—-]?\s*([^\n\.]+)',
            r'^([A-Z][^\.]{10,200}\.)$',  # Простые предложения как возможные подписи
        ]
        
        candidates = []
        
        for block in text.get("blocks", []):
            if block.get("type") != 0:
                continue
            block_bbox = block.get("bbox")
            if not block_bbox:
                continue
            
            # Проверка близости по вертикали
            block_y = (block_bbox[1] + block_bbox[3]) / 2
            elem_y = (y0 + y1) / 2
            
            if abs(block_y - elem_y) > max_distance:
                continue
            
            # Извлекаем текст блока
            block_text = " ".join(
                span.get("text", "").strip()
                for line in block.get("lines", [])
                for span in line.get("spans", [])
            ).strip()
            
            if not block_text or len(block_text) > 300:
                continue
            
            # Проверка на паттерны подписей
            for pattern in caption_patterns:
                match = re.search(pattern, block_text, re.IGNORECASE | re.MULTILINE)
                if match:
                    caption = match.group(1).strip() if match.lastindex else block_text
                    candidates.append((abs(block_y - elem_y), caption))
                    break
        
        # Возвращаем ближайшую подходящую подпись
        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1]
        
        return ""
    
    def _get_context(self, text_dict: Dict, span: Dict, max_words: int = 10) -> str:
        """Получает контекст вокруг span"""
        context_words = []
        target_bbox = span["bbox"]
        
        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for s in line.get("spans", []):
                    s_bbox = s["bbox"]
                    # Проверка на ту же строку (по вертикали)
                    if abs(s_bbox[1] - target_bbox[1]) < 5:
                        words = s.get("text", "").split()
                        context_words.extend(words)
        
        # Возвращаем слова до и после (упрощённо)
        return " ".join(context_words[:max_words])
    
    def _build_page_text(self, text_dict: Dict, formulas: List[Formula], 
                        table_bboxes: List[Tuple], exclude_bboxes: List[Tuple]) -> str:
        """Сборка текста страницы с заменой формул и таблиц на маркеры"""
        lines = []
        
        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
                
            for line in block.get("lines", []):
                line_parts = []
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue
                    
                    span_bbox = tuple(span["bbox"])
                    
                    # Пропускаем если это часть формулы или таблицы
                    if self._bbox_overlaps_any(span_bbox, exclude_bboxes):
                        continue
                    
                    line_parts.append(text)
                
                if line_parts:
                    lines.append(" ".join(line_parts))
        
        # Добавляем маркеры для формул
        for f in formulas:
            lines.append(f"«ФОРМУЛА {f.formula_type}: ${f.latex}$»")
        
        return "\n".join(lines)
    
    def _bbox_overlaps_any(self, bbox: Tuple, bbox_list: List[Tuple], threshold: float = 0.5) -> bool:
        """Проверяет перекрытие bbox с любым из списка"""
        if not bbox or not bbox_list:
            return False
        x0, y0, x1, y1 = bbox
        
        for other in bbox_list:
            ox0, oy0, ox1, oy1 = other
            # Простая проверка перекрытия
            if not (x1 < ox0 or x0 > ox1 or y1 < oy0 or y0 > oy1):
                # Вычисляем площадь перекрытия
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