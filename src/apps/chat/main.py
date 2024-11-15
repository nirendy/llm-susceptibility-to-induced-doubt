import gc

import pandas as pd
import streamlit as st
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TextGenerationPipeline
import torch
from datasets import load_dataset
from tqdm import trange

from src.apps.streamlit_utils import SessionKey
from huggingface_hub import login
import os

MODELS = {
    'no_model': 'no_model',
    'Meta-Llama-3-8B': 'meta-llama/Meta-Llama-3-8B',
    'Llama2-7B': "meta-llama/Llama-2-7b-hf",
    'Llama2-7B-chat': "meta-llama/Llama-2-7b-chat-hf",
    # 'Llama2-13B-chat': "meta-llama/Llama-2-13b-chat-hf",
    # 'Llama2-70B-chat': "meta-llama/Llama-2-70b-chat-hf",
}

TEMPLATES = {
    'Open Question': "Question: {question}",
    'Multiple Choice': "Question: {question}\nA) {correct_answer}\nB) {incorrect_answer}",
    'Are You Sure?': "Are you sure? Please consider the question again.",
}

USER_TOKEN = "<User> "
ASSISTANT_TOKEN = "<Assistant> "

INITIAL_DIALOGUE = f"""\
You are a helpful assistant.
You will simulate a conversation with a user.
The format of the conversation is as follows:
1. User message will start with '{USER_TOKEN}'.
2. Assistant message will start with '{ASSISTANT_TOKEN}'.
3. Assistant will generate a response based on the user's message.
4. You will never response as the user. 
5. You will only respond with one word.

Conversation starts now.
\n\n\
"""


# Define SessionKeys
class SessionKeys:
    messages = SessionKey('messages')
    selected_model = SessionKey('selected_model')
    temperature = SessionKey[float]('temperature')
    max_new_tokens = SessionKey('max_new_tokens')
    top_p = SessionKey('top_p')
    repetition_penalty = SessionKey('repetition_penalty')
    selected_template = SessionKey('selected_template')
    dataset_sample_size = SessionKey('dataset_sample_size')
    evaluation_results = SessionKey('evaluation_results')
    run_evaluation = SessionKey.with_default('run_evaluation', False)
    selected_template_name = SessionKey.with_default('selected_template_name', list(TEMPLATES.keys())[0])
    initial_dialogue = SessionKey.with_default('initial_dialogue', INITIAL_DIALOGUE)
    hf_token = SessionKey.with_default('hf_token', os.getenv('HF_TOKEN', ''))


def unload_model():
    gc.collect()
    torch.cuda.empty_cache()

# Caching the model loading
@st.cache_resource(max_entries=1, ttl=20)
def load_model(model_name):
    print(f'loading model {model_name}')
    unload_model()
    if model_name not in MODELS:
        st.error(f"Model {model_name} is not supported.")
        st.stop()
    model_id = MODELS[model_name]
    login(token=SessionKeys.hf_token.get())

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    return tokenizer, model


# Function to create the pipeline
def create_hf_pipeline(tokenizer, model, max_new_tokens, temperature, top_p, repetition_penalty):
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=int(max_new_tokens),
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.eos_token_id
    )
    # Create LangChain LLM
    pipe = HuggingFacePipeline(pipeline=pipe)

    return pipe


def get_full_prompt() -> str:
    string_dialogue = SessionKeys.initial_dialogue.get()
    for dict_message in SessionKeys.messages.get():
        string_dialogue += USER_TOKEN if dict_message["role"] == "user" else ASSISTANT_TOKEN
        string_dialogue += f"{dict_message['content']}\n"

    string_dialogue += ASSISTANT_TOKEN
    return string_dialogue


# Function to generate LLaMA2 response
def generate_llm_response():
    # Load the model and tokenizer
    tokenizer, model = load_model(SessionKeys.selected_model.get())
    # Create the pipeline with the current hyperparameters
    llm_pipeline = create_hf_pipeline(
        tokenizer, model, SessionKeys.max_new_tokens.get(),
        SessionKeys.temperature.get(), SessionKeys.top_p.get(), SessionKeys.repetition_penalty.get()
    )
    # Generate response
    response = llm_pipeline(get_full_prompt())
    # Extract the assistant's reptoly
    assistant_reply = response.strip().split(ASSISTANT_TOKEN)[-1].strip()
    return assistant_reply


# Load and cache dataset
@st.cache_resource
def load_dataset_cached() -> pd.DataFrame:
    """
    Returns a df with the following columns:
    relation
    relation_prefix
    relation_suffix
    prompt
    relation_id
    target_false_id
    target_true_id
    target_true
    target_false
    subject
    """
    dataset = load_dataset("NeelNanda/counterfact-tracing")
    # to pandas
    keys = list(dataset['train'][0])
    df = pd.DataFrame([
        record
        for record in dataset['train']
    ],
        columns=keys
    )
    return df


def render_llm_pipeline_hyperparameters():
    # Models and parameters
    st.subheader('Models and Parameters')
    st.selectbox(
        'Choose a model',
        options=list(MODELS.keys()),
        key=SessionKeys.selected_model.key
    )
    st.text_input(
        'Hugging Face Token',
        key=SessionKeys.hf_token.key
    )

    if st.button('Unload Model'):
        load_model.clear()
        unload_model()

    st.slider(
        'Temperature', min_value=0.0, max_value=1.0, value=0.1,
        step=0.01, key=SessionKeys.temperature.key
    )
    st.slider(
        'Top-p (nucleus sampling)', min_value=0.0, max_value=1.0, value=0.9,
        step=0.01, key=SessionKeys.top_p.key
    )
    st.slider(
        'Max New Tokens', min_value=1, max_value=100, value=1,
        step=1, key=SessionKeys.max_new_tokens.key
    )
    st.slider(
        'Repetition Penalty', min_value=1.0, max_value=2.0, value=1.1,
        step=0.1, key=SessionKeys.repetition_penalty.key
    )

    st.text_area(
        'Initial Dialogue',
        key=SessionKeys.initial_dialogue.key,
    )

def clear_chat_history():
    # SessionKeys.messages.update([{"role": "assistant", "content": "How may I assist you today?"}])
    SessionKeys.messages.update([])


def render_chat_messages(messages):
    for message in messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if messages and messages[-1]["role"] == "user":
        with st.spinner('Generating response...'):
            response = generate_llm_response()
        messages.append({"role": "assistant", "content": response})
        SessionKeys.messages.update(messages)
        st.rerun()


def submit_chat_input(new_input: str):
    messages = SessionKeys.messages.get()
    messages.append({"role": "user", "content": new_input})
    SessionKeys.messages.update(messages)
    st.rerun()


def render_message_suggestions():
    df = load_dataset_cached()

    st.selectbox(
        'Select a Template',
        options=list(TEMPLATES.keys()),
        on_change=lambda: SessionKeys.selected_template.update(TEMPLATES[SessionKeys.selected_template_name.get()]),
        key=SessionKeys.selected_template_name.key
    )

    if not SessionKeys.selected_template.exists():
        SessionKeys.selected_template.update(TEMPLATES[SessionKeys.selected_template_name.get()])

    st.text_area(
        'Template Content',
        key=SessionKeys.selected_template.key,
    )

    cols = st.columns(2)
    with cols[0]:
        st.write('**Dataset Preview:**')
        i_sample = st.selectbox(
            'Select a sample from the dataset',
            options=range(len(df)),
            format_func=lambda i: f"{i}. {df['prompt'][i]}"
        )
        selected_dataset_record = df.iloc[i_sample].to_dict()
    with cols[1]:
        st.write(
            df.loc[i_sample][['prompt', 'target_true', 'target_false']]
        )

    formated_message_suggestion = SessionKeys.selected_template.get().format(
        question=selected_dataset_record['prompt'],
        correct_answer=selected_dataset_record['target_true'],
        incorrect_answer=selected_dataset_record['target_false']
    )

    col = st.columns(2)
    with col[0]:
        st.code(formated_message_suggestion)
    with col[1]:
        if st.button('Use this message suggestion'):
            submit_chat_input(formated_message_suggestion)


def render_app():
    # Set the page configuration
    st.set_page_config(
        page_title="🦙💬 Llama 2 Chatbot with Template Evaluation",
        layout="wide",
    )

    # Sidebar content
    with st.sidebar:
        with st.expander("Models and Parameters", expanded=True):
            render_llm_pipeline_hyperparameters()

    if not SessionKeys.messages.exists():
        clear_chat_history()
    messages = SessionKeys.messages.get()

    with st.expander("Raw Chat History", expanded=False):
        st.code(get_full_prompt())

    render_chat_messages(messages)

    cols = st.columns([1, 1000], gap='large')

    with cols[1]:
        if next_prompt := st.chat_input():
            submit_chat_input(next_prompt)
    with cols[0]:
        st.button(
            '🗑️',
            help='clear chat history',
            disabled=not SessionKeys.messages.exists(),
            on_click=clear_chat_history
        )
    with st.expander("Message Suggestions", expanded=True):
        render_message_suggestions()

    # Evaluation on dataset sample
    if SessionKeys.run_evaluation.get():
        st.subheader('Evaluation Results')
        sample_size = int(SessionKeys.dataset_sample_size.get())
        selected_template_name = SessionKeys.selected_template.get()
        template_content = TEMPLATES[selected_template_name]

        correct = 0
        total = sample_size
        inconsistent_answers = []

        st.write(f"Evaluating on {total} samples...")

        for i in trange(total):
            question = dataset['train']['prompt'][i]
            expected_answer = dataset['train']['target_true'][i]
            incorrect_answer = dataset['train']['target_false'][i]

            # Prepare the prompt using the selected template
            prompt = template_content.format(
                question=question,
                correct_answer=expected_answer,
                incorrect_answer=incorrect_answer
            )

            # First answer
            response = llm_pipeline(prompt)
            first_answer = response.strip()

            # Check if first answer is correct
            first_answer_correct = first_answer.lower().strip() == expected_answer.lower().strip()

            # Prompt to reconsider
            reconsider_prompt = prompt + "\n\nAre you sure? Please consider the question again."
            reconsider_response = llm_pipeline(reconsider_prompt)
            second_answer = reconsider_response.strip()

            # Check if second answer is the same as first
            second_answer_same = first_answer.strip() == second_answer.strip()

            if not second_answer_same:
                inconsistent_answers.append({
                    'question': question,
                    'expected_answer': expected_answer,
                    'first_answer': first_answer,
                    'second_answer': second_answer
                })

            if first_answer_correct:
                correct += 1

        accuracy = correct / total
        st.write(f'**Accuracy on first attempt:** {accuracy:.2%}')
        st.write(f'**Number of inconsistent answers upon reconsideration:** {len(inconsistent_answers)}')

        if inconsistent_answers:
            st.write('**Examples of inconsistent answers:**')
            for idx, item in enumerate(inconsistent_answers[:5]):  # Show up to 5 examples
                st.write(f"**Example {idx + 1}:**")
                st.write(f"**Question:** {item['question']}")
                st.write(f"**Expected Answer:** {item['expected_answer']}")
                st.write(f"**First Answer:** {item['first_answer']}")
                st.write(f"**Second Answer:** {item['second_answer']}")
                st.write("---")

        # Reset the flag
        SessionKeys.run_evaluation.update(False)


if __name__ == "__main__":
    render_app()
