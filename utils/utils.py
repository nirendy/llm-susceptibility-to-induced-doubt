import torch


def generate_a_or_b(
    pipeline,
    question: str,
    correct_answer: str,
    incorrect_answer: str,
    sample_answer: bool = False,
    max_length: int = 2,
    prefix: str = ""
):
    # Construct a prompt that encourages a short answer
    if torch.rand(1) < 0.5:
        answer = "a"
        full_prompt = f"Which answer is correct? a or b:\n\nQuestion: {question}\n\na: {correct_answer}\nb: {incorrect_answer}\n"
    else:
        answer = "b"
        full_prompt = f"Which answer is correct? a or b:\n\nQuestion: {question}\n\na: {incorrect_answer}\nb: {correct_answer}\n"
    
    full_prompt = prefix + full_prompt
    # Generate the output
    output = pipeline(full_prompt, max_new_tokens=max_length, num_return_sequences=1, 
                       temperature=0.7, top_p=0.95, do_sample=sample_answer)
    
    # Extract and return the generated answer
    generated_text = output[0]['generated_text']
    predicted_answer = generated_text.split("<Assistant>")[-1].strip()
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
    predicted_answer = generated_text.split("<Assistant>")[-1].strip()
    answer_correct = predicted_answer == correct_answer
    return answer_correct, generated_text