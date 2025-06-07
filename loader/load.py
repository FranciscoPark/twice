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
    og_dataset = []

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
                    "content": f"Please translate the following English text into {language}, ensuring that every part, including the 'Context:' label and all options (A, B, C, D), is translated without omission. Maintain the original formatting and structure. Do not change the answer index and the symbol. Include Context: \n\n{prompt}"
                }
            ]
        )

        trans_prompt = trans_response.choices[0].message.content

        local_example = """Original 1. Context: A group of friends on a road trip in California see a sign for In-N-Out. They've heard great things about it. They excitedly pull into the drive-thru. The first thing they do is
    A. ask for a menu of their sushi options.
    B. look for the "Animal Style" options on the secret menu or ask for a Double-Double.
    C. try to order a bucket of fried chicken.
    D.complain that there are no vegetarian Szechuan dishes.

    Localized 1. Context : 친구들이 차를 타고 가다 맘스터치 간판을 봅니다. 다들 맛집으로 유명하다는 이야기를 들었습니다. 설레는 마음으로 드라이브 스루로 들어갑니다. 그들이 제일 먼저 하는 일은: 
A. 메뉴판을 보며 김치찌개를 시키려 합니다. 
B. 불고기 버거 세트나 싸이버거 세트를 주문합니다. 
C. 족발 하나를 주문하려고 합니다. 
D. 비건 메뉴가 없다고 투덜거립니다.

Original 2. Context: A high school student is getting ready for prom night. Her date is due to arrive any minute. She quickly
A. changes into her pajamas.
B. puts on her corsage that's been chilling in the fridge.
C. starts studying for her math exam.
D. decides to wash her car.

Localized 2. Context: 한 고등학생이 수학여행 장기자랑 무대에 오를 준비를 하고 있습니다. 곧 자기 차례가 다가옵니다. 그는 빠르게: 
A. 잠옷으로 갈아입습니다. 
B. 미리 준비한 댄스 곡에 맞춰 최종 안무를 연습합니다. 
C. 다음날 볼 수학 시험 공부를 시작합니다. 
D. 교실 청소를 하기로 결정합니다.
"""
        
        # # localizing the translated prompt using GPT-4o
        # local_response = client.chat.completions.create(
        #     model="gpt-4o",
        #     messages=[
        #         {
        #             "role": "user",
        #             "content": f"Revise the sentences below into a sentences that align with the culture of {language}-speaking regiions. Do not change {language} itself ,the answer index, and the symbol. Just localize it.\n\n{trans_prompt}"
        #         }
        #     ]
        # )

        # localizing the translated prompt using GPT-4o

        # give local example
        local_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": f"""Please localize the following text into {language}. Ensure that cultural references are adapted to resonate with a {language} audience, modifying elements such as food items, idioms, commonsense knowledge, famous brands, and cultural practices to their {language} equivalents. Ensure that every part, including the '맥락:' or 'Context:' labels and all options (A, B, C, D), is localized without omission. Give me only the localized text, not examples. \n\n{trans_prompt} You can check the example below. \n\n{local_example}"""
                }
            ]
        )

        # do not give local example
        # local_response = client.chat.completions.create(
        #     model="gpt-4o",
        #     messages=[
        #         {
        #             "role": "user",
        #             "content": f"""Please localize the following text into {language}. Ensure that cultural references are adapted to resonate with a {language} audience, modifying elements such as food items, idioms, commonsense knowledge, famous brands, and cultural practices to their {language} equivalents. Ensure that every part, including the '맥락:' or 'Context:' labels and all options (A, B, C, D), is localized without omission. \n\n{trans_prompt}"""
        #         }
        #     ]
        # )


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
        og_dataset.append([prompt, trans_prompt, prompt_tok] )
        answer_list.append(data["answer"] if "answer" in data else None)
        pbar.update(1)

    # Sort by input length for efficient batching
    dataset.sort(key=lambda x: len(x[2]), reverse=True)
    og_dataset.sort(key=lambda x: len(x[-1]), reverse=True)

    print(f"dataset loaded for {name}-single-{num_shot} shot-seed{fewshot_seed}")
    return dataset, answer_list, og_dataset
