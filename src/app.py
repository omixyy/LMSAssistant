from core import DocumentProcessor
from core.grading.grader import Grader
from core.grading.inquirer import Inquirer
from core.llm.ollama_client import OllamaClient
from core.rubrics.rubric_generator import RubricGenerator
from core.grading.models import Question, QAPair
from core.rag.retriever import QARetriever
from core.rag.vector_store import VectorStore

from chromadb import PersistentClient

options = {
    "num_ctx": 64000,      # много контекста: методичка + ответ + рубрика
    "num_predict": 4096,   # хватает на подробный JSON-отчёт
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

grader = Grader(ollama_client)

# Обработка PDF
student_answer, _ = processor.process(
    '../test_files/main.pdf',
)

student_task, _ = processor.process(
    '../test_files/Методические указания к лабораторной работе 1.01.pdf',
)

# Возможность преподавателем изменять рубрику
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

chroma_client = PersistentClient(path='./chroma_db')
qa_retriever = QARetriever(client=chroma_client, collection_name='qa_pairs')
materials_store = VectorStore(client=chroma_client, collection_name="teaching_materials")

inquirer = Inquirer(analysis)
questions = inquirer.generate_questions()

# materials_store.add_documents(
#     ids=["guidelines_lab1", "sample_report_lab1"],
#     texts=[
#         open("docs/guidelines_lab1.txt", encoding="utf-8").read(),
#         open("docs/sample_report_lab1.txt", encoding="utf-8").read(),
#     ],
#     metadatas=[
#         {"type": "guidelines", "lab": "1.01"},
#         {"type": "sample_report", "lab": "1.01"},
#     ],
# )

# print(analysis)

# print(inquirer.get_unconfident_criteria())

qa_pairs = []
print(questions)
for question in questions:
    print(str(question))
    qa_pairs.append(QAPair(question=question, answer=input()))

qa_retriever.add_pairs(qa_pairs)
print(qa_retriever.query(questions[0].text, top_k=1, rubric_item_id=questions[0].related_rubric_item_id))