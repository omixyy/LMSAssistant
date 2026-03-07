import re
from typing import List, Dict, Any, Optional


class TechnicalValidator:
    """Проверка технических требований к работе"""

    def __init__(self):
        self.required_sections = [
            "введение",
            "цель работы",
            "теоретическая часть",
            "практическая часть",
            "выводы",
            "список литературы"
        ]

        self.gost_requirements = {
            "min_chars_per_page": 1500,  # Минимум символов на странице для ГОСТ
            "min_sources": 5,  # Минимум источников в списке литературы
            "max_paragraph_length": 5000,  # Максимальная длина абзаца
        }

    def validate(self, text: str, metadata: Optional[Dict] = None) -> List[str]:
        """
        Проверка работы и возврат списка замечаний
        """
        issues = []
        text_lower = text.lower()

        # Проверка наличия обязательных разделов
        issues.extend(self._check_sections(text_lower))

        # Проверка структуры
        issues.extend(self._check_structure(text))

        # Проверка списка литературы
        issues.extend(self._check_references(text_lower))

        # Проверка объема
        if metadata:
            issues.extend(self._check_volume(metadata))

        return issues

    def _check_sections(self, text_lower: str) -> List[str]:
        """Проверка наличия обязательных разделов"""
        issues = []
        found_sections = []

        for section in self.required_sections:
            if section in text_lower:
                found_sections.append(section)
            else:
                # Проверяем синонимы
                synonyms = self._get_section_synonyms(section)
                found = any(syn in text_lower for syn in synonyms)

                if found:
                    found_sections.append(section)
                else:
                    issues.append(f"Отсутствует раздел '{section}'")

        return issues

    def _get_section_synonyms(self, section: str) -> List[str]:
        """Получение синонимов для разделов"""
        synonyms_map = {
            "введение": ["введ"],
            "цель работы": ["целью", "целями", "цель"],
            "теоретическая часть": ["теоретич", "теория"],
            "практическая часть": ["практич", "эксперимент", "реализация"],
            "выводы": ["заключени", "вывод"],
            "список литературы": ["литература", "источники", "библиографи"]
        }
        return synonyms_map.get(section, [])

    def _check_structure(self, text: str) -> List[str]:
        """Проверка структуры документа"""
        issues = []

        # Разбивка на абзацы
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        # Проверка длины абзацев
        for i, para in enumerate(paragraphs):
            if len(para) > self.gost_requirements["max_paragraph_length"]:
                issues.append(f"Абзац {i + 1} слишком длинный (>5000 символов)")

        # Проверка наличия списков (маркированных или нумерованных)
        if not re.search(r'[•\-*\d]+\.', text):
            issues.append("Отсутствуют маркированные или нумерованные списки")

        return issues

    def _check_references(self, text_lower: str) -> List[str]:
        """Проверка списка литературы"""
        issues = []

        # Поиск раздела с литературой
        lit_index = -1
        for pattern in ["список литературы", "литература", "источники"]:
            idx = text_lower.find(pattern)
            if idx != -1:
                lit_index = idx
                break

        if lit_index == -1:
            issues.append("Не найден раздел 'Список литературы'")
            return issues

        # Подсчет источников (простая эвристика)
        lit_section = text_lower[lit_index:lit_index + 2000]

        # Поиск паттернов, похожих на ссылки
        source_patterns = [
            r'\d+\.\s+[а-яёa-z]+',  # "1. Иванов"
            r'\[[\d\s,]+\]',  # [1] или [1,2]
            r'\([\d]{4}\)',  # (2023)
            r'http[s]?://',  # URL
        ]

        sources_count = 0
        for pattern in source_patterns:
            matches = re.findall(pattern, lit_section, re.IGNORECASE)
            sources_count += len(matches)

        if sources_count < self.gost_requirements["min_sources"]:
            issues.append(f"Список литературы содержит менее {self.gost_requirements['min_sources']} источников")

        return issues

    def _check_volume(self, metadata: Dict) -> List[str]:
        """Проверка объема работы"""
        issues = []

        # Проверка на слишком маленький объем
        word_count = metadata.get("word_count", 0)
        if word_count < 500:
            issues.append(f"Работа слишком короткая ({word_count} слов)")

        # Проверка на подозрительно большой объем
        if word_count > 10000:
            issues.append(f"Работа очень большая ({word_count} слов). Проверьте на наличие лишнего материала")

        return issues