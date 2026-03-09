from core.parser.enhanced_parser import EnhancedFileProcessor
from core.prompter.prompt_builder import PromptBuilder

import ollama


processor = EnhancedFileProcessor(
    extract_images=True,
    detect_tables=True,
)

prompt_builder = PromptBuilder()

# Обработка PDF
student_answer, _ = processor.process(
    'report.pdf',
)

student_task, _ = processor.process(
    'Методические указания к лабораторной работе 1.01.pdf',
)

# Простой запрос
def ask_gemma(model_name, prompt):
    optimal_config = {
        'model': model_name,
        'prompt': prompt,  # prompter передается отдельно, не внутри options
        'options': {
            'num_ctx': 32000,  # Хороший баланс скорость/качество
            'num_predict': 2048,  # Максимальная длина ответа
            'temperature': 0.3,  # Для точности
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

llm_response = ask_gemma(
    'deepseek-v3.1:671b-cloud',
    llm_prompt,
)

print(llm_response)
