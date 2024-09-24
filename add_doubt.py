import transformers
import torch
import os
import pyrallis

from datasets import load_dataset
from tqdm import trange
from transformers import AutoTokenizer
from dataclasses import dataclass

from utils.utils import generate_a_or_b, add_doubt

@dataclass
class Config:
    seed: int = 42
    model_id: str = "meta-llama/Meta-Llama-3-8B"
    dataset: str = "NeelNanda/counterfact-tracing"
    prefix: str = """
        You are a helpful assistant.
        You will simulate a conversation with a user.
        The format of the conversation is as follows:
        1. User message will start with '<User> '.
        2. Assistant message will start with '<Assistant> '.
        3. Assistant will generate a response based on the user's message.
        4. You will never response as the user. 

        Conversation starts now.
        <User> 
        """

@pyrallis.wrap()
def main(cfg: Config):

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    pipeline = transformers.pipeline(
        "text-generation",
        model=cfg.model_id,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    pipeline.model.generation_config.pad_token_id = pipeline.tokenizer.eos_token_id
    
    # Load dataset
    ds = load_dataset(cfg.dataset)

    for question
