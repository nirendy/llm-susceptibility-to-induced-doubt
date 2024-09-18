import streamlit as st
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from src.apps.streamlit_utils import SessionKey


# Define SessionKeys
class SessionKeys:
    messages = SessionKey("messages")
    selected_model = SessionKey("selected_model")
    temperature = SessionKey("temperature")
    max_new_tokens = SessionKey("max_new_tokens")
    top_p = SessionKey("top_p")
    repetition_penalty = SessionKey("repetition_penalty")


# Set the page configuration
st.set_page_config(page_title="🦙💬 Llama 2 Chatbot")

# Sidebar content
with st.sidebar:
    st.title('🦙💬 Llama 2 Chatbot')

    # Models and parameters
    st.subheader('Models and Parameters')
    st.selectbox(
        'Choose a Llama2 model',
        ['Llama2-7B', 'Llama2-13B', 'Llama2-70B'],
        key=SessionKeys.selected_model.key
    )
    st.slider(
        'Temperature', min_value=0.0, max_value=1.0, value=0.1,
        step=0.01, key=SessionKeys.temperature.key
    )
    st.slider(
        'Top-p (nucleus sampling)', min_value=0.0, max_value=1.0, value=0.9,
        step=0.01, key=SessionKeys.top_p.key
    )
    st.slider(
        'Max New Tokens', min_value=1, max_value=2048, value=1,
        step=8, key=SessionKeys.max_new_tokens.key
    )

    st.slider(
        'Repetition Penalty', min_value=1.0, max_value=2.0, value=1.1,
        step=0.1, key=SessionKeys.repetition_penalty.key
    )


    # Button to clear chat history
    def clear_chat_history():
        SessionKeys.messages.update([{"role": "assistant", "content": "How may I assist you today?"}])


    st.button('Clear Chat History', on_click=clear_chat_history)

# Initialize messages if not already initialized
SessionKeys.messages.init([{"role": "assistant", "content": "How may I assist you today?"}])
messages = SessionKeys.messages.get()

# Display chat messages
for message in messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# Caching the model loading
@st.cache_resource
def load_model(model_name):
    model_id = ""
    if model_name == 'Llama2-7B':
        model_id = "meta-llama/Llama-2-7b-chat-hf"
    elif model_name == 'Llama2-13B':
        model_id = "meta-llama/Llama-2-13b-chat-hf"
    elif model_name == 'Llama2-70B':
        model_id = "meta-llama/Llama-2-70b-chat-hf"
    else:
        st.error(f"Model {model_name} is not supported.")
        st.stop()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    return tokenizer, model


# Function to create the pipeline
def create_pipeline(tokenizer, model, max_new_tokens, temperature, top_p, repetition_penalty):
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    return pipe


# Load the model and tokenizer
selected_model = SessionKeys.selected_model.get()
tokenizer, model = load_model(selected_model)

# Get the current hyperparameters
max_new_tokens = SessionKeys.max_new_tokens.get()
temperature = SessionKeys.temperature.get()
top_p = SessionKeys.top_p.get()
repetition_penalty = SessionKeys.repetition_penalty.get()

# Create the pipeline with the current hyperparameters
pipe = create_pipeline(tokenizer, model, max_new_tokens, temperature, top_p, repetition_penalty)

# Create LangChain LLM
local_llm = HuggingFacePipeline(pipeline=pipe)


# Function to generate LLaMA2 response
def generate_llama2_response(prompt_input):
    string_dialogue = "You are a helpful assistant.\n\n"
    for dict_message in messages:
        if dict_message["role"] == "user":
            string_dialogue += f"User: {dict_message['content']}\n"
        else:
            string_dialogue += f"Assistant: {dict_message['content']}\n"
    string_dialogue += f"User: {prompt_input}\nAssistant:"

    # Generate response
    response = local_llm(string_dialogue)
    # Extract the assistant's reply
    assistant_reply = response.strip().split("Assistant:")[-1].strip()
    return assistant_reply


# User-provided prompt
if prompt := st.chat_input():
    # Append user message to session state
    messages.append({"role": "user", "content": prompt})
    SessionKeys.messages.update(messages)

    with st.chat_message("user"):
        st.write(prompt)

    # Generate assistant's response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_llama2_response(prompt)
            st.write(response)
        # Append assistant's response to session state
        messages.append({"role": "assistant", "content": response})
        SessionKeys.messages.update(messages)
