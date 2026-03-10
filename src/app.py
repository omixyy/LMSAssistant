from core import DocumentProcessor
from core.grading.grader import Grader
from core.grading.inquirer import Inquirer
from core.llm.ollama_client import OllamaClient
from core.rubrics.rubric_generator import RubricGenerator
from core.prompting.prompt_builder import PromptBuilder


options = {
    "num_ctx": 64000,      # много контекста: методичка + ответ + рубрика
    "num_predict": 2048,   # хватает на подробный JSON-отчёт
    "temperature": 0.2,    # максимальная детерминированность
    "top_p": 0.8,          # слегка сужаем выбор токенов
    "repeat_penalty": 1.1, # меньше повторов в тексте
    "seed": 42,            # фиксированный сид, если модель/ollama поддерживают
}

ollama_client = OllamaClient('deepseek-v3.1:671b-cloud', options)

processor = DocumentProcessor(
    extract_images=True,
    detect_tables=True,
    detect_formulas=True,
)

rubric_generator = RubricGenerator(ollama_client)

prompt_builder = PromptBuilder()

grader = Grader(ollama_client)

# Обработка PDF
student_answer, _ = processor.process(
    '../test_files/main.pdf',
)

student_task, _ = processor.process(
    '../test_files/Методические указания к лабораторной работе 1.01.pdf',
)

rubric = (
    rubric_generator.generate_from_text(
        "lab_1",
        "Рубрика проверки лабораторной работы №1",
        student_task,
    )
)

# prompt = prompt_builder.build_prompt(
#     student_task,
#     student_answer,
#     rubric=rubric,
#     use_cot=True,
# )

# print(prompt)

# print(rubric)

analysis = grader.grade(student_task, student_answer, rubric)
inquirer = Inquirer(analysis)
print(analysis)

print(inquirer.get_unconfident_criteria())
