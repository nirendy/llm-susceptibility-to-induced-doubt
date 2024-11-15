import json
from typing import Callable
import transformers
import torch
import pyrallis
from enum import Enum

from tqdm import trange, tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass
from collections import defaultdict
import os
from huggingface_hub import login
from src.types import DATASETS, MODEL_ARCH, DatasetArgs, TModelID
from src.consts import MODEL_SIZES_PER_ARCH_TO_MODEL_ID, PATHS
from utils.utils import format_prompt
from src.utils.slurm import submit_job
from pathlib import Path
from src.datasets.download_dataset import load_custom_dataset

# login(token=os.getenv("HUGGINGFACE_TOKEN"))

# "microsoft/Phi-3.5-mini-instruct"
# meta-llama/Llama-3.2-1B
# meta-llama/Llama-3.2-3B
# meta-llama/Llama-3.1-8B
# mistralai/Mixtral-8x7B-v0.1
# mistralai/Mistral-Nemo-Instruct-2407


@dataclass
class Config:
    seed: int = 42
    model_arch: MODEL_ARCH = MODEL_ARCH.LLAMA3_2
    model_size: str = "1B"
    dataset: str = pyrallis.field(
        default=DatasetArgs(name=DATASETS.COUNTER_FACT, splits="train2"),
        is_mutable=True,
    )
    question_frasing = """User: I have a question with two possible answers. Please provide the correct one, either a or b.
    Question: {question}
    a: {first_answer}
    b: {second_answer}
    Instructions: Please choose the correct answer from (a) or (b).
    
    Answer:"""

    experiment_name: str = ""
    doubt_phrase = "\nUser: I am not sure about the answer. Can you try again? Just answer with a or b\n\nAnswer:"
    output_path = PATHS.OUTPUT_DIR / "add_doubt_logits_diff"
    with_slurm: bool = False

    @property
    def model_id(self) -> TModelID:
        return MODEL_SIZES_PER_ARCH_TO_MODEL_ID[self.model_arch][self.model_size]


def main_local(cfg: Config):
    print(cfg)
    assert cfg.experiment_name, "Please provide an experiment name"
    cfg.output_path.mkdir(parents=True, exist_ok=True)
    result_file = cfg.output_path / f"{cfg.experiment_name}.json"

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"Model {cfg.model_id} loaded")

    # Load dataset
    ds = load_custom_dataset(cfg.dataset)
    pbar = tqdm(enumerate(ds), desc="Processing", total=ds.shape[0])
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
            
            non_spaced_correct_answer = "a" if correct_first else "b"
            non_spaced_wrong_answer = "b" if correct_first else "a"

            # Get token IDs for 'a' and 'b'
            # Add space before to ensure correct tokenization
            # TODO: Check if this is necessary
            correct_token_id = tokenizer.encode(correct_answer)[1]
            wrong_token_id = tokenizer.encode(wrong_answer)[1]
            non_spaced_correct_token_id = tokenizer.encode(non_spaced_correct_answer)[1]
            non_spaced_wrong_token_id = tokenizer.encode(non_spaced_wrong_answer)[1]

            answer = correct_answer if correct_response else wrong_answer
            rest_of_sentence = answer + cfg.doubt_phrase

            format_q_tokens = tokenizer(formatted_q, return_tensors="pt")
            rest_of_sentence_tokens = tokenizer(rest_of_sentence, return_tensors="pt")

            inputs_ids = torch.cat(
                [
                    format_q_tokens["input_ids"],
                    rest_of_sentence_tokens["input_ids"],
                ],
                dim=-1,
            )
            inputs_ids = inputs_ids.to(model.device)
            first_generated_location = format_q_tokens["input_ids"].shape[-1]

            # Predict Next token
            with torch.no_grad():
                outputs = model(inputs_ids)

            logits = outputs.logits

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
            json.dump(convert_results_for_json(results), result_file.open("w"))

    # Convert results to DataFrame for analysis
    json.dump(
        convert_results_for_json(results),
        result_file.open("w"),
    )
    return results


# Convert tuple keys to strings when saving
def convert_results_for_json(results):
    return {
        f"{correct_first}_{correct_response}": value
        for (correct_first, correct_response), value in results.items()
    }


@pyrallis.wrap()
def main(cfg: Config):
    if cfg.with_slurm:
        gpu_type = "a100"
        # gpu_type = "titan_xp-studentrun"

        for model_arch, model_size, gpus in [
            (MODEL_ARCH.LLAMA2, "8B", 1),
            (MODEL_ARCH.LLAMA3_1, "8B", 1),
            (MODEL_ARCH.LLAMA3_2, "1B", 1),
            (MODEL_ARCH.LLAMA3_2, "3B", 1),
            (MODEL_ARCH.MISTRAL, "8x7B", 1),
            (MODEL_ARCH.MISTRAL, "Nemo", 1),
            (MODEL_ARCH.PHI, "3.5-mini", 1),
        ]:
            cfg.model_arch = model_arch
            cfg.model_size = model_size

            job_name = f"{model_arch}_{model_size}_{cfg.dataset.dataset_name}"
            cfg.experiment_name = job_name

            job = submit_job(
                main_local,
                cfg,
                log_folder=str(
                    PATHS.SLURM_DIR / "add_doubt_logits_diff" / job_name / "%j"
                ),
                job_name=job_name,
                # timeout_min=1200,
                gpu_type=gpu_type,
                slurm_gpus_per_node=gpus,
            )

            print(f"{job}: {job_name}")
    else:
        
        cfg.experiment_name = f"test_{cfg.model_arch}_{cfg.model_size}_{cfg.dataset.dataset_name}"
        main_local(cfg)


if __name__ == "__main__":
    main()
