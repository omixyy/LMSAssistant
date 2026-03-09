from core import DocumentProcessor
from core.prompter.prompt_builder import PromptBuilder

import ollama


processor = DocumentProcessor(
    extract_images=True,
    detect_tables=True,
    detect_formulas=True
)

prompt_builder = PromptBuilder()

# Обработка PDF
student_answer, _ = processor.process(
    'main.pdf',
)

student_task, _ = processor.process(
    'Методические указания к лабораторной работе 1.01.pdf',
)

# Простой запрос
def ask_llm(model_name, prompt):
    optimal_config = {
        'model': model_name,
        'prompt': prompt,
        'options': {
            'num_ctx': 64000,
            'num_predict': 2048,
            'temperature': 0.3,
        }
    }

    response = ollama.generate(**optimal_config)

    # Для generate() ответ находится в response['response']
    result_text = response['response']
    return result_text


llm_prompt = prompt_builder.build_prompt(
    student_task,
    student_answer,
)

print(llm_prompt)

llm_response = ask_llm(
    'deepseek-v3.1:671b-cloud',
    llm_prompt,
)

print(llm_response)
