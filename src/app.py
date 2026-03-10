from core import DocumentProcessor
from core.grading.grader import Grader
from core.llm.ollama_client import OllamaClient


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

# Обработка PDF
student_answer, _ = processor.process(
    '../test_files/main.pdf',
)

student_task, _ = processor.process(
    '../test_files/Методические указания к лабораторной работе 1.01.pdf',
)

grader = Grader(ollama_client)
analysis = grader.grade(student_task, student_answer)

print(analysis.total_score)

# prompt_builder = PromptBuilder()

# prompt = prompt_builder.build_prompt(
#     student_task,
#     student_answer,
# )

# print(prompt)

# response = ollama_client.generate(prompt)

# print(response)

