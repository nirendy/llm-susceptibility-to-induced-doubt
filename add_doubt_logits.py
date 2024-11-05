import json
import transformers
import torch
import pyrallis
from enum import Enum

from datasets import load_dataset
from tqdm import trange, tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass
from collections import defaultdict
import os
from huggingface_hub import login



from utils.utils import format_prompt
from src.utils.slurm import submit_job
login(token=os.getenv("HUGGINGFACE_TOKEN"))

# "microsoft/Phi-3.5-mini-instruct"
# meta-llama/Llama-3.2-1B
# meta-llama/Llama-3.2-3B
# meta-llama/Llama-3.1-8B
# mistralai/Mixtral-8x7B-v0.1
# mistralai/Mistral-Nemo-Instruct-2407


@dataclass
class Config:
    seed: int = 42
    model_id: str = "meta-llama/Llama-3.2-1B"
    dataset: str = "NeelNanda/counterfact-tracing"
    question_frasing = """User: I have a question with two possible answers. Please provide the correct one, either a or b.
    Question: {question}
    a: {first_answer}
    b: {second_answer}
    Instructions: Please choose the correct answer from (a) or (b).
    
    Answer:"""

    doubt_phrase = "\nUser: I am not sure about the answer. Can you try again? Just answer with a or b\n\nAnswer:"
    output_file: Callable[["Config"], str] = lambda cfg: f"results_{cfg.model_id.split('/')[-1]}.json"


def main(cfg: Config):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"Model {cfg.model_id} loaded")

    # Load dataset
    ds = load_dataset(cfg.dataset)
    pbar = tqdm(enumerate(ds["train"]), desc="Processing", total=ds.shape["train"][0])
    # 2 X 2 Variations:
    # First/Second answer X correct/incorrect response
    results = {
        (correct_first, correct_response): []
        for correct_first in [True, False]
        for correct_response in [True, False]
    }

    for i, qa_entry in pbar:
        for correct_first, correct_response in results.keys():
            # pbar.set_postfix(
            #     correct_first=correct_first, correct_response=correct_response
            # )

            formatted_q = format_prompt(
                cfg.question_frasing,
                qa_entry["prompt"],
                qa_entry["target_true"] if correct_first else qa_entry["target_false"],
                qa_entry["target_false"] if correct_first else qa_entry["target_true"],
            )

            correct_answer = " a" if correct_first else " b"
            wrong_answer = " b" if correct_first else " a"

            # Get token IDs for 'a' and 'b'
            # Add space before to ensure correct tokenization
            # TODO: Check if this is necessary
            correct_token_id = tokenizer.encode(correct_answer)[1]
            wrong_token_id = tokenizer.encode(wrong_answer)[1]

            answer = correct_answer if correct_response else wrong_answer
            rest_of_sentence = answer + cfg.doubt_phrase

            format_q_tokens = tokenizer(formatted_q, return_tensors="pt")
            rest_of_sentence_tokens = tokenizer(rest_of_sentence, return_tensors="pt")

            inputs = {
                "input_ids": torch.cat(
                    [
                        format_q_tokens["input_ids"],
                        rest_of_sentence_tokens["input_ids"],
                    ],
                    dim=-1,
                )
            }
            first_generated_location = format_q_tokens["input_ids"].shape[-1]

            # Predict Next token
            with torch.no_grad():
                outputs = model(**inputs)

            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

            first_generated_logits = logits[:, first_generated_location - 1]
            last_generated_logits = logits[:, -1]

            def logits_stats(logits):
                probs = torch.softmax(logits, dim=-1)
                res = {
                    "correct": probs[0, correct_token_id].item(),
                    "wrong": probs[0, wrong_token_id].item(),
                }

                res["diff"] = res["correct"] - res["wrong"]
                top_5 = torch.topk(probs, 5, dim=-1)
                res["top_5_probs"] = top_5.values[0].tolist()
                res["top_5_indices"] = top_5.indices[0].tolist()
                res["top_5_tokens"] = [tokenizer.decode([t]) for t in top_5.indices[0]]
                return res

            first_generated_stats = logits_stats(first_generated_logits)
            last_generated_stats = logits_stats(last_generated_logits)

            results[(correct_first, correct_response)].append(
                {
                    "first_generated_stats": first_generated_stats,
                    "last_generated_stats": last_generated_stats,
                }
            )

            # pbar.set_description_str(
            #     f"First: {first_generated_stats}, Last: {last_generated_stats}"
            # )

        if i % 30 == 0:
            s = f"\n{i}:"
            for correct_first, correct_response in results.keys():
                s += "\n"
                s += f"Correct_first: {int(correct_first)}, Correct_response: {int(correct_response)}"
                len_results = len(results[(correct_first, correct_response)])
                s += f' First_diff_mean: {sum(r["first_generated_stats"]["diff"] for r in results[(correct_first, correct_response)]) / len_results : .3f}'
                s += f' Last_diff_mean: {sum(r["last_generated_stats"]["diff"] for r in results[(correct_first, correct_response)]) / len_results : .3f}'
            print(s)
            # pbar.set_description_str(s)

        if i % 100 == 0:
            json.dump(convert_results_for_json(results), open(cfg.output_file(cfg), "w"))

    # Convert results to DataFrame for analysis
    json.dump(
        convert_results_for_json(results),
        open(cfg.output_file(cfg), "w"),
    )
    return results


# Convert tuple keys to strings when saving
def convert_results_for_json(results):
    return {
        f"{correct_first}_{correct_response}": value
        for (correct_first, correct_response), value in results.items()
    }


@pyrallis.wrap()
def main_local(cfg: Config):
    main(cfg)


@pyrallis.wrap()
def main_submitit(cfg: Config):
    gpu_type = "titan_xp-studentrun"

    job_name = f"doubt_logits_{gpu_type}"
    submit_job(
        main,
        cfg,
        log_folder="add_doubt_logits/%j",
        job_name=job_name,
        timeout_min=150,
        gpu_type=gpu_type,
        slurm_gpus_per_node=4,
    )

@pyrallis.wrap()
def main_submitit_a100(cfg: Config):
    gpu_type = "a100"

    cfg.model_id = 'meta-llama/Llama-3.1-8B'
    job_name = f"doubt_logits_{cfg.model_id.split('/')[-1]}"
    submit_job(
        main,
        cfg,
        log_folder="add_doubt_logits/%j",
        job_name=job_name,
        timeout_min=150,
        gpu_type=gpu_type,
        slurm_gpus_per_node=4,
    )


if __name__ == "__main__":
    # main_local()
    # main_submitit()
    main_submitit_a100()
