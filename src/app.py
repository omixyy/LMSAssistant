from core import DocumentProcessor
from core.grading.grader import Grader
from core.llm.ollama_client import OllamaClient
from core.rubrics.rubric_generator import RubricGenerator
from core.prompting.prompt_builder import PromptBuilder


options = {
    'num_ctx': 64000,
    'num_predict': 2048,
    'temperature': 0.3,
}

ollama_client = OllamaClient('deepseek-v3.1:671b-cloud', options)

processor = DocumentProcessor(
    extract_images=True,
    detect_tables=True,
    detect_formulas=True,
)

rubric_generator = RubricGenerator(ollama_client)

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

print(rubric)

analysis = grader.grade(student_task, student_answer, rubric)

print(analysis)
