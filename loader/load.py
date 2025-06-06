from datasets import load_dataset, Dataset
from tqdm import tqdm
import random
from loader.load_utils import preprocess_map, get_fewshot_single, get_fewshot_mixture, get_batched
from transformers import PreTrainedTokenizer
from loader.task_config import get_task_config
import string
from typing import Optional, Tuple, List, Any
from openai import OpenAI, OpenAIError
import os

def load(name: str,
        tokenizer: PreTrainedTokenizer,
        instruction: bool = True,
        cand_type: str = "large", 
        num_shot: int = 0,
        fewshot_seed: int = 1234,
        language: str = "Korean",
        limit: Optional[int] = None) -> Tuple[List[List[Any]], List[Optional[int]]]:
    """
    Load and preprocess datasets for evaluation under the new rules:
      1) If config.test_with_labels=True, use config.split as test set.
         Otherwise, use the original validation set as test set.
      2) If test_with_labels=False, also do an 80:20 split of the original train set
         -> new_train_data (80%) and new_val_data(20%). The new_val_data is used 
            for few-shot sampling.
      3) If config.has_dev=True, we prefer to sample few-shot from dev. 
         If dev is smaller than num_shot, we then sample remainder from 
         the new validation or official validation set as needed.
      4) Evaluate on the final test set. 
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")

    client = OpenAI(api_key=api_key)

    config = get_task_config(name)

    # -------------------------
    # 1) Decide on the final test set
    # -------------------------
    if config.test_with_labels:
        # e.g. normal: raw_data = test set
        test_split = config.split
        raw_data = load_dataset(
            path=config.path, 
            name=config.name, 
            split=test_split,
            trust_remote_code=True
        )
    else:
        # Use the original validation set as the new "test set"
        raw_data = load_dataset(
            path=config.path, 
            name=config.name, 
            split="validation",
            trust_remote_code=True
        )
        # Additionally, for future fine-tuning or few-shot, we do an 80:20 split from the original train
        train_data = load_dataset(
            path=config.path,
            name=config.name,
            split="train",
            trust_remote_code=True
        )
        # Shuffle and split 80:20
        train_data = train_data.shuffle(seed=42)
        split_idx = int(len(train_data) * config.split_ratio)
        config.new_train_data = train_data.select(range(split_idx))
        config.new_train_data = preprocess_map(config.new_train_data, name=name) # newly added
        config.new_val_data = train_data.select(range(split_idx, len(train_data)))
        config.new_val_data = preprocess_map(config.new_val_data, name=name) # newly added

    # Preprocess the final test dataset
    raw_data = preprocess_map(raw_data, name=name)
    print(f"Loading the raw data of {name} is done!")

    if not limit:
        limit = len(raw_data)
    limit = min(limit, len(raw_data))

    dataset = []
    answer_list = []

    # -------------------------
    # 2) Build prompts for each example in test set
    # -------------------------
    
    # Add few-shot examples if requested
    if num_shot > 0:
        fewshot = get_fewshot_single(
            name=name,
            num_shot=num_shot,
            seed=fewshot_seed,
            cand_type=cand_type
        )
    else:
        fewshot = ''

    
    
    pbar = tqdm(total=limit, desc=f"Translating and localizing {name} into {language}..")

    for i in range(limit):
        data = raw_data[i]

        # Possibly add instructions
        if instruction:
            prefix = config.get_instruction("single")
        else:
            prefix = ""

        prefix += fewshot

        # Build the question + choices
        if config.question_name:
            real_question = f'{config.question_name}: {data["question"]}\n'
        else:
            real_question = data["question"] + "\n"
        for j, choice_text in enumerate(data["choices"]):
            if cand_type == "large":
                real_question += f'{string.ascii_uppercase[j]}. {choice_text}\n'
            else:
                real_question += f'{string.ascii_lowercase[j]}. {choice_text}\n'

        real_question += f'{config.completion}:'

        prompt = prefix + real_question
        
        # translation using GPT-4o
        trans_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": f"Translate the sentences below to {language}. Do not change the answer index and the symbol.\n\n{prompt}"
                }
            ]
        )

        trans_prompt = trans_response.choices[0].message.content
        
        # localizing the translated prompt using GPT-4o
        local_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": f"Revise the sentences below into a sentences that align with the culture of {language}-speaking regiions. Do not change {language} itself ,the answer index, and the symbol. Just localize it.\n\n{trans_prompt}"
                }
            ]
        )

        local_prompt = local_response.choices[0].message.content
        prompt_tok = tokenizer(local_prompt, add_special_tokens=True)["input_ids"]

        # We'll build both uppercase/lowercase candidate tokens
        cand_list = [
            [f' {string.ascii_uppercase[j]}' for j in range(len(data["choices"]))],
            [f' {string.ascii_lowercase[j]}' for j in range(len(data["choices"]))],
        ]
        cand_list_tok = []
        for group in cand_list:
            cand_list_tok.append([
                tokenizer(a, add_special_tokens=False)["input_ids"][0] for a in group
            ])

        dataset.append([local_prompt, cand_list, prompt_tok, cand_list_tok, i])
        answer_list.append(data["answer"] if "answer" in data else None)
        pbar.update(1)

    # Sort by input length for efficient batching
    dataset.sort(key=lambda x: len(x[2]), reverse=True)

    print(f"dataset loaded for {name}-single-{num_shot} shot-seed{fewshot_seed}")
    return dataset, answer_list
