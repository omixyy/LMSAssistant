import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import ReportReviewer, GradingRubric


def main():
    # Создание рецензента
    reviewer = ReportReviewer(
        model_name="gemma3:4b",
        rubric=GradingRubric.default_technical()
    )

    # Путь к файлу для проверки
    file_path = "sample_report.pdf"  # Замените на ваш файл

    if not os.path.exists(file_path):
        print(f"Файл не найден: {file_path}")
        print("Создайте тестовый файл или укажите правильный путь")
        return

    try:
        # Запуск проверки
        print("Начинаю проверку работы...")
        result = reviewer.review(
            file_path=file_path,
            task_path='description.pdf',
            assignment_description="Отчет по лабораторной работе №1",
            use_reflection=False,  # Использовать саморефлексию
        )

        # Вывод результатов
        print(result)

        # Сохранение в JSON
        json_output = result.to_json()
        with open("review_result.json", "w", encoding="utf-8") as f:
            f.write(json_output)
        print("\nРезультат сохранен в review_result.json")

    except Exception as e:
        print(f"Ошибка при проверке: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()