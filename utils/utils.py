import torch
import string

def format_prompt(
    question_frasing: str,
    question: str,
    first_answer: str,
    second_answer: str,
    prefix: str = "",
):
    """
    Formats a prompt with placeholders for the question and answers.
    Args:
        question_frasing (str): The template for phrasing the question, containing placeholders for answers and the question itself.
        question (str): The question to be asked.
        correct_answer (str): The correct answer to the question.
        incorrect_answer (str): The incorrect answer to the question.
        prefix (str, optional): A prefix to add to the prompt. Defaults to "".
    Returns:
        str: The formatted prompt.
    """
    
    return prefix + question_frasing.replace(
        "{first_answer}", first_answer
    ).replace("{second_answer}", second_answer).replace("{question}", question)
    

def generate_a_or_b(
    pipeline,
    question_frasing: str,
    question: str,
    correct_answer: str,
    incorrect_answer: str,
    sample_answer: bool = False,
    max_length: int = 2,
    prefix: str = "",
):
    """
    Generates a prompt and evaluates the model's response to determine if it matches the expected answer.
    Args:
        pipeline: The language model pipeline used for generating responses.
        question_frasing (str): The template for phrasing the question, containing placeholders for answers and the question itself.
        question (str): The question to be asked.
        correct_answer (str): The correct answer to the question.
        incorrect_answer (str): The incorrect answer to the question.
        sample_answer (bool, optional): Whether to sample the answer. Defaults to False.
        max_length (int, optional): The maximum length of the generated response. Defaults to 2.
        prefix (str, optional): A prefix to add to the prompt. Defaults to "".
    Returns:
        tuple: A tuple containing:
            - answer_correct (bool): Whether the model's predicted answer matches the expected answer.
            - answer (str): The expected answer ('a' or 'b').
            - generated_text (str): The full generated text from the model.
    """
    # Construct a prompt that encourages a short answer
    if torch.rand(1) < 0.5:
        answer = "a"
        full_prompt = format_prompt(question_frasing, question, correct_answer, incorrect_answer, prefix)
    else:
        answer = "b"
        full_prompt = format_prompt(question_frasing, question, incorrect_answer, correct_answer, prefix)

    full_prompt = prefix + full_prompt
    # Generate the output
    output = pipeline(full_prompt, max_new_tokens=max_length, num_return_sequences=1, 
                       temperature=0.7, top_p=0.95, do_sample=sample_answer)
    
    # Extract and return the generated answer
    generated_text = output[0]['generated_text']
    predicted_answer = generated_text.split("Answer:")[-1].strip().translate(str.maketrans('', '', string.punctuation))

    answer_correct = predicted_answer == answer
    
    return answer_correct, answer, generated_text


def add_doubt(
    pipeline,
    prefix: str,
    doubt_phrase: str,
    correct_answer: str,
    sample_answer: bool = False,
    max_length: int = 2
):
    full_prompt = prefix + doubt_phrase
    
    output = pipeline(full_prompt, max_new_tokens=max_length, num_return_sequences=1, 
                       temperature=0.7, top_p=0.95, do_sample=sample_answer)
    
    # Extract and return the generated answer
    generated_text = output[0]['generated_text']
    predicted_answer = generated_text.split("Answer:")[-1].strip().translate(str.maketrans('', '', string.punctuation))
    answer_correct = predicted_answer == correct_answer
    return answer_correct, generated_text
