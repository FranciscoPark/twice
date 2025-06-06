from loader.task_config import get_task_config, SUBTASK_TO_PARENT, KMMLU_subject, MMLU_subject
import random
from datasets import load_dataset, Dataset
import random
from transformers import PreTrainedTokenizer, PreTrainedModel
from typing import List, Union
import torch
import string
from collections.abc import Iterator
from collections import defaultdict

def preprocess_map(raw_data : Dataset, name : str) -> Dataset:
    """goal : all data have three columns : question(str), choices(list[str]), answer(int)"""

    # Get configuration to check if this is a subtask
    config = get_task_config(name)
    
    # For subtasks of multi-task benchmarks, use parent preprocessing
    if hasattr(config, 'parent_task'):
        parent_name = SUBTASK_TO_PARENT.get(name)
        if parent_name:
            # Use parent task's preprocessing logic
            return preprocess_multitask(raw_data, parent_name, name)
    
    # Original preprocessing logic for single tasks
    """=========answer part========="""
    if config.answer:

        # sanity check for answer
        valid_values = []
        invalid_values = []
        

        for val in set(raw_data[config.answer]):

            if isinstance(val, str) and val.isdigit():
                val = int(val)

            if isinstance(val, int):
                valid_values.append(val)
            elif isinstance(val, str) and val.isalnum() and len(val) == 1:
                valid_values.append(val)
            else:
                invalid_values.append(val)
        
        alpha_values = sorted([v for v in valid_values if isinstance(v, str)])
        alpha_mapping = {v: i for i, v in enumerate(alpha_values)}

        int_values = sorted([v for v in valid_values if isinstance(v, int)])
        int_mapping = {v: i for i, v in enumerate(int_values)}

        value_mapping = {**alpha_mapping, **int_mapping}

        mapped_answers = []

        for val in raw_data[config.answer]:

            if isinstance(val, str) and val.isdigit():
                val = int(val)
            mapped_answers.append(value_mapping.get(val, val))  # maintain dirty data
        
        if invalid_values:
            print("Invalid (dirty) values found:", invalid_values)    


        raw_data = raw_data.remove_columns(config.answer)
        raw_data = raw_data.add_column("answer", mapped_answers)




    """=========question and choices part==========="""
    new_choices = []
    new_answers = []

    if name in ["ko_arc_easy", "ko_arc_challenge", "arc_easy", "arc_challenge"]:
        new_choices = [item["text"] for item in raw_data["choices"]]
        raw_data = raw_data.remove_columns("choices")
        raw_data = raw_data.add_column("choices", new_choices)
    
    elif name=="ko_winogrande":
        raw_data = raw_data.rename_column("sentence", "question")

        new_choices = [[opt1, opt2] for opt1, opt2 in zip(raw_data["option1"], raw_data["option2"])]
        raw_data = raw_data.add_column("choices", new_choices)

    elif name=="snu_lambada": # answer is None
        raw_data = raw_data.rename_column("text", "question")

        rng = random.Random(128)

        for ans, cand in zip(raw_data["answer"], raw_data["candidate"]):
            options = [ans, cand]

            rng.shuffle(options)
            new_choices.append(options)
            new_answers.append(options.index(ans))

        raw_data = raw_data.remove_columns(["answer", "candidate"])
        raw_data = raw_data.add_column("choices", new_choices)
        raw_data = raw_data.add_column("answer", new_answers)        


    elif name=="ko_hellaswag":
        raw_data = raw_data.rename_column("context", "question")
        
        new_choices = [[end1, end2, end3, end4] for end1, end2, end3, end4 in 
                      zip(raw_data["ending_1"], raw_data["ending_2"], raw_data["ending_3"], raw_data["ending_4"])]

        raw_data = raw_data.add_column("choices", new_choices)

    elif name=="hellaswag":
        import re
        
        def _preprocess(text: str):
            text = text.strip()
            # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
            text = text.replace(" [title]", ". ")
            text = re.sub("\\[.*?\\]", "", text)
            text = text.replace("  ", " ")
            return text            
        
        capitalized_b_list = [b.capitalize() for b in raw_data["ctx_b"]]
        temp_questions = [ctx_a + " " + ctx_b for ctx_a, ctx_b in zip(raw_data["ctx_a"], capitalized_b_list)]
        new_questions = [_preprocess(each) for each in temp_questions]

        raw_data = raw_data.add_column("question", new_questions)
        raw_data = raw_data.rename_column("endings", "choices")
    
    elif name=="piqa": # answer is None
        raw_data = raw_data.rename_column("goal", "question")

        new_choices = []
        new_answers = []
        labels = raw_data["label"]

        for sol1, sol2, lab in zip(raw_data["sol1"], raw_data["sol2"], labels):
            if lab == 0:
                correct_sol = sol1
                incorrect_sol = sol2
            else:
                correct_sol = sol2
                incorrect_sol = sol1
            
            options = [correct_sol, incorrect_sol]
            random.shuffle(options)
            answer_index = options.index(correct_sol)
            
            new_choices.append(options)
            new_answers.append(answer_index)


        raw_data = raw_data.add_column("choices", new_choices)
        raw_data = raw_data.add_column("answer", new_answers)

    elif name=="truthful_qa":
        new_choices = [each["mc1_targets"]["choices"] for each in raw_data]
        new_answers = [each["mc1_targets"]["labels"].index(1) for each in raw_data]

        raw_data = raw_data.add_column("choices", new_choices)
        raw_data = raw_data.add_column("answer", new_answers)
    
    elif name=="winogrande":
        raw_data = raw_data.rename_column("sentence", "question")
        new_choices = [[opt1, opt2] for opt1, opt2 in zip(raw_data["option1"], raw_data["option2"])]

        raw_data = raw_data.add_column("choices", new_choices)
    
    elif name=="openbookqa":
        raw_data = raw_data.rename_column("question_stem", "question")
        new_choices = [each["choices"]["text"] for each in raw_data]

        raw_data = raw_data.remove_columns("choices")
        raw_data = raw_data.add_column("choices", new_choices)

    elif name == "commonsense_qa":
            new_choices = [each["choices"]["text"] for each in raw_data]

            raw_data = raw_data.remove_columns("choices")
            raw_data = raw_data.add_column("choices", new_choices)

    elif name == "boolq":   
        new_questions = []
        new_choices = []
        new_answers = []

        for row in raw_data:
            question_text = row['question']
            if question_text:
                question_text = question_text[0].upper() + question_text[1:]
                if not question_text.endswith('?'):
                    question_text += '?'
                            
            combined_q = f"Passage: {row['passage']}\n\nQuestion: {question_text}"
            new_questions.append(combined_q)
            new_choices.append(["Yes", "No"])
            new_answers.append(0 if row["answer"] else 1)

        raw_data = raw_data.remove_columns(["question", "answer", "passage"])
        raw_data = raw_data.add_column("question", new_questions)
        raw_data = raw_data.add_column("choices", new_choices)
        raw_data = raw_data.add_column("answer", new_answers)

    return raw_data


def preprocess_multitask(raw_data: Dataset, parent_name: str, subtask_name: str) -> Dataset:
    """Preprocessing for multi-task benchmarks like MMLU and KMMLU"""
    
    if parent_name == "kmmlu":
        # KMMLU has choices as A, B, C, D columns
        new_choices = [[A, B, C, D] for A, B, C, D in 
                      zip(raw_data["A"], raw_data["B"], raw_data["C"], raw_data["D"])]
        raw_data = raw_data.add_column("choices", new_choices)
    
    elif parent_name == "mmlu":
        # MMLU already has the right format
        pass
    
    # Add more multi-task preprocessing as needed
    
    return raw_data

def get_fewshot_single(name: str, num_shot: int=5, seed: int=1234, cand_type: str="large") -> str:
    """
    Returns a few-shot prompt from either:
      1) dev set (if config.has_dev=True),
         if num_shot > dev_size, sample remainder from validation set or new_val_data
      2) validation set or new_val_data (if has_dev=False)
         - new_val_data is used if config.test_with_labels=False
         - official validation set if config.test_with_labels=True
    """
    config = get_task_config(name)
    random.seed(seed)

    # 1) Attempt to load dev set if config.has_dev is True
    dev_data = None
    val_data = None

    if config.has_dev:
        # Load dev
        dev_data = load_dataset(
            path=config.path,
            name=config.name,
            split=config.fewshot_split,  # e.g. "dev"
            trust_remote_code=True
        )
        dev_data = preprocess_map(dev_data, name=name)

    # For sampling from validation:
    # - If test_with_labels=True, use official "validation" split.
    # - If test_with_labels=False, use config.new_val_data (the 20% splitted from train).
    if config.test_with_labels:
        # official val
        val_data = load_dataset(
            path=config.path,
            name=config.name,
            split="validation",
            trust_remote_code=True
        )
        val_data = preprocess_map(val_data, name=name)
    else:
        # newly splitted val
        val_data = config.new_val_data  # set in load_single() if needed

    fewshot_indices = []
    fewshot_prompt = ""

    # Now figure out how many from dev vs. val
    if dev_data is not None:
        dev_size = len(dev_data)
        if num_shot <= dev_size:
            # All from dev
            indices = random.sample(range(dev_size), num_shot)
            fewshot_prompt = _build_fewshot_prompt(dev_data, indices, cand_type, name)
            return fewshot_prompt
        else:
            # Use entire dev
            indices_all = list(range(dev_size))
            fewshot_prompt = _build_fewshot_prompt(dev_data, indices_all, cand_type, name)
            leftover = num_shot - dev_size

            # Then sample leftover from val_data
            if val_data is None or len(val_data) == 0:
                # No fallback
                return fewshot_prompt  # best we can do

            if leftover > len(val_data):
                leftover = len(val_data)  # clamp

            val_indices = random.sample(range(len(val_data)), leftover)
            fewshot_prompt += _build_fewshot_prompt(val_data, val_indices, cand_type, name)
            return fewshot_prompt

    else:
        # dev_data is None => sample from val_data only
        if val_data is None or len(val_data) == 0:
            return ""  # no data at all

        if num_shot > len(val_data):
            num_shot = len(val_data)  # clamp

        indices = random.sample(range(len(val_data)), num_shot)
        fewshot_prompt = _build_fewshot_prompt(val_data, indices, cand_type, name)
        return fewshot_prompt


def _build_fewshot_prompt(data: Dataset, indices: List[int], cand_type: str, task_name: str) -> str:
    """
    Build the textual prompt for each shot in indices.
    """
    config = get_task_config(task_name)
    prompt_str = ""
    for i in indices:
        shot = data[i]
        if config.question_name:
            prompt_str += f'{config.question_name}: {shot["question"]}\n'
        else:
            prompt_str += shot["question"] + "\n"
        for j, choice_text in enumerate(shot["choices"]):
            if cand_type == "large":
                prompt_str += f'{string.ascii_uppercase[j]}. {choice_text}\n'
            else:
                prompt_str += f'{string.ascii_lowercase[j]}. {choice_text}\n'

        # Provide the correct choice
        if "answer" in shot and shot["answer"] is not None:
            ans_idx = shot["answer"]
            if cand_type == "large":
                prompt_str += f'{config.completion}: {string.ascii_uppercase[ans_idx]}\n\n'
            else:
                prompt_str += f'{config.completion}: {string.ascii_lowercase[ans_idx]}\n\n'
        else:
            # If there's no gold label (corner case),
            # we just skip specifying an answer or do something minimal
            prompt_str += f'{config.completion}: \n\n'
    return prompt_str

def get_fewshot_mixture(
    name: str,
    num_shot: int = 5,
    seed: int = 1234,
    cand_type: str = "large"
) -> str:
    """
    Same logic as get_fewshot_single, except we append the full choice text
    in the answer line. i.e., "Answer: A. ChoiceText"
    """

    config = get_task_config(name)
    random.seed(seed)

    dev_data = None
    val_data = None

    # 1) If the task has a dedicated dev split:
    if config.has_dev:
        # Load dev
        dev_data = load_dataset(
            path=config.path,
            name=config.name,
            split=config.fewshot_split,  # e.g. "dev"
            trust_remote_code=True
        )
        dev_data = preprocess_map(dev_data, name=name)

    # 2) Figure out which set to use for validation:
    #    If test_with_labels=True, use official "validation".
    #    Otherwise, use config.new_val_data (80:20 split from train).
    if config.test_with_labels:
        val_data = load_dataset(
            path=config.path,
            name=config.name,
            split="validation",
            trust_remote_code=True
        )
        val_data = preprocess_map(val_data, name=name)
    else:
        # new_val_data was set during load_single(...) if test_with_labels=False
        val_data = config.new_val_data

    # 3) Construct few-shot prompt
    fewshot_prompt = ""
    if dev_data is not None:
        dev_size = len(dev_data)
        if num_shot <= dev_size:
            # All from dev
            indices = random.sample(range(dev_size), num_shot)
            fewshot_prompt = _build_fewshot_prompt_mixture(dev_data, indices, cand_type, name)
            return fewshot_prompt
        else:
            # Use entire dev
            indices_all = list(range(dev_size))
            fewshot_prompt = _build_fewshot_prompt_mixture(dev_data, indices_all, cand_type, name)
            leftover = num_shot - dev_size

            if val_data is None or len(val_data) == 0:
                # No fallback
                return fewshot_prompt

            if leftover > len(val_data):
                leftover = len(val_data)

            val_indices = random.sample(range(len(val_data)), leftover)
            fewshot_prompt += _build_fewshot_prompt_mixture(val_data, val_indices, cand_type, name)
            return fewshot_prompt

    else:
        # No dedicated dev -> sample all from val_data
        if val_data is None or len(val_data) == 0:
            return ""  # no data at all
        if num_shot > len(val_data):
            num_shot = len(val_data)

        indices = random.sample(range(len(val_data)), num_shot)
        fewshot_prompt = _build_fewshot_prompt_mixture(val_data, indices, cand_type, name)
        return fewshot_prompt


def _build_fewshot_prompt_mixture(
    data: Dataset,
    indices: List[int],
    cand_type: str,
    task_name: str
) -> str:
    """
    Builds a few-shot prompt in 'mixture' style, i.e., including the full choice
    text in the answer line: "Answer: A. <Full text>"
    """
    config = get_task_config(task_name)
    prompt_str = ""
    for i in indices:
        shot = data[i]
        if config.question_name:
            prompt_str += f'{config.question_name}: {shot["question"]}\n'
        else:
            prompt_str += shot["question"] + "\n"
        for j, choice_text in enumerate(shot["choices"]):
            if cand_type == "large":
                prompt_str += f'{string.ascii_uppercase[j]}. {choice_text}\n'
            else:
                prompt_str += f'{string.ascii_lowercase[j]}. {choice_text}\n'

        # Provide the correct choice + the full text
        if "answer" in shot and shot["answer"] is not None:
            ans_idx = shot["answer"]
            if cand_type == "large":
                prompt_str += (
                    f'{config.completion}: '
                    f'{string.ascii_uppercase[ans_idx]}. {shot["choices"][ans_idx]}\n\n'
                )
            else:
                prompt_str += (
                    f'{config.completion}: '
                    f'{string.ascii_lowercase[ans_idx]}. {shot["choices"][ans_idx]}\n\n'
                )
        else:
            # If no gold label is provided, we can either omit or do something minimal
            prompt_str += f'{config.completion}: \n\n'
    return prompt_str

def get_fewshot_log(
    name: str,
    num_shot: int = 5,
    seed: int = 1234,
    cand_type: str = "large"
) -> str:
    """
    Returns a few-shot prompt for the 'log' setting from either:
      1) A dev split (if config.has_dev=True).
         If num_shot > dev_size, sample remainder from val/new_val_data.
      2) Otherwise, from validation or newly splitted validation data 
         (depending on config.test_with_labels).
    
    Unlike 'single' or 'mixture', we do not label choices as A/B/C/D in
    the few-shot. We just present the question and the *correct* completion
    as free-form text, so that the model can see an example of how to answer
    “open-ended” style. 
    """
    config = get_task_config(name)
    random.seed(seed)

    dev_data = None
    val_data = None

    # 1) If the task has a dedicated dev split
    if config.has_dev:
        dev_data = load_dataset(
            path=config.path,
            name=config.name,
            split=config.fewshot_split,  # e.g. "dev"
            trust_remote_code=True
        )
        dev_data = preprocess_map(dev_data, name=name)

    # 2) For validation or fallback:
    if config.test_with_labels:
        # Official validation set
        val_data = load_dataset(
            path=config.path,
            name=config.name,
            split="validation",
            trust_remote_code=True
        )
        val_data = preprocess_map(val_data, name=name)
    else:
        # Use the newly split 20% validation from the original train
        val_data = config.new_val_data

    # 3) Actually sample
    if dev_data is not None:
        dev_size = len(dev_data)
        if num_shot <= dev_size:
            indices = random.sample(range(dev_size), num_shot)
            return _build_fewshot_prompt_log(dev_data, indices, cand_type, name)
        else:
            # Use entire dev, then remainder from val
            if dev_size == 0:
                # Edge case: empty dev
                pass
            indices_all = list(range(dev_size))
            fewshot_prompt = _build_fewshot_prompt_log(dev_data, indices_all, cand_type, name)
            leftover = num_shot - dev_size
            if val_data is None or len(val_data) == 0:
                # No fallback
                return fewshot_prompt

            # clamp leftover
            leftover = min(leftover, len(val_data))
            val_indices = random.sample(range(len(val_data)), leftover)
            fewshot_prompt += _build_fewshot_prompt_log(val_data, val_indices, cand_type, name)
            return fewshot_prompt
    else:
        # No dev set => sample purely from val_data
        if not val_data or len(val_data) == 0:
            return ""
        if num_shot > len(val_data):
            num_shot = len(val_data)
        indices = random.sample(range(len(val_data)), num_shot)
        return _build_fewshot_prompt_log(val_data, indices, cand_type, name)


def _build_fewshot_prompt_log(
    data: Dataset,
    indices: List[int],
    cand_type: str,
    task_name: str
) -> str:
    """
    For 'log' mode, we do not label choices as "A, B, C, D". 
    Instead we show the question and the correct completion as plain text.

    Example format per shot:
      Question: <question text>
      Answer: <the correct choice's text>

    Then a blank line.
    """
    config = get_task_config(task_name)
    prompt_str = ""
    for i in indices:
        shot = data[i]

        prompt_str += shot["question"] + " "

        # Show the *correct* choice text (if known)
        if "answer" in shot and shot["answer"] is not None:
            ans_idx = shot["answer"]
            correct_text = shot["choices"][ans_idx]
            prompt_str += correct_text + "\n\n"
        else:
            # If no gold label is present, just leave it blank or do something minimal
            prompt_str += f"\n\n"

    return prompt_str

def get_fewshot_special(
    name: str,
    num_shot: int = 5,
    seed: int = 1234,
    cand_type: str = "large"
) -> str:
    """
    Returns a few-shot prompt for the 'special' setting (used for winogrande-style tasks).
    This is for tasks where we have a sentence with a blank ("_") that needs to be filled.
    
    The few-shot examples show the sentence with the blank filled by the correct answer.
    
    Similar to other fewshot functions:
      1) If config.has_dev=True, sample from dev set first.
         If num_shot > dev_size, sample remainder from val/new_val_data.
      2) Otherwise, sample from validation or newly split validation data 
         (depending on config.test_with_labels).
    """
    config = get_task_config(name)
    random.seed(seed)

    dev_data = None
    val_data = None

    # 1) If the task has a dedicated dev split
    if config.has_dev:
        dev_data = load_dataset(
            path=config.path,
            name=config.name,
            split=config.fewshot_split,  # e.g. "dev"
            trust_remote_code=True
        )
        dev_data = preprocess_map(dev_data, name=name)

    # 2) For validation or fallback:
    if config.test_with_labels:
        # Official validation set
        val_data = load_dataset(
            path=config.path,
            name=config.name,
            split="validation",
            trust_remote_code=True
        )
        val_data = preprocess_map(val_data, name=name)
    else:
        # Use the newly split 20% validation from the original train
        val_data = config.new_val_data

    # 3) Actually sample
    if dev_data is not None:
        dev_size = len(dev_data)
        if num_shot <= dev_size:
            indices = random.sample(range(dev_size), num_shot)
            return _build_fewshot_prompt_special(dev_data, indices, cand_type, name)
        else:
            # Use entire dev, then remainder from val
            indices_all = list(range(dev_size))
            fewshot_prompt = _build_fewshot_prompt_special(dev_data, indices_all, cand_type, name)
            leftover = num_shot - dev_size
            
            if val_data is None or len(val_data) == 0:
                # No fallback
                return fewshot_prompt

            # clamp leftover
            leftover = min(leftover, len(val_data))
            val_indices = random.sample(range(len(val_data)), leftover)
            fewshot_prompt += _build_fewshot_prompt_special(val_data, val_indices, cand_type, name)
            return fewshot_prompt
    else:
        # No dev set => sample purely from val_data
        if not val_data or len(val_data) == 0:
            return ""
        if num_shot > len(val_data):
            num_shot = len(val_data)
        indices = random.sample(range(len(val_data)), num_shot)
        return _build_fewshot_prompt_special(val_data, indices, cand_type, name)


def _build_fewshot_prompt_special(
    data: Dataset,
    indices: List[int],
    cand_type: str,
    task_name: str
) -> str:
    """
    For 'special' mode (winogrande-style), we show the sentence with the blank filled
    by the correct answer.
    
    Format per shot:
      <sentence with blank filled by correct answer>
      
    Then a blank line.
    """
    config = get_task_config(task_name)
    prompt_str = ""
    
    for i in indices:
        shot = data[i]
        
        # Get the sentence with blank
        sentence_with_blank = shot["question"]
        
        # Fill the blank with the correct answer
        if "answer" in shot and shot["answer"] is not None:
            ans_idx = shot["answer"]
            correct_choice = shot["choices"][ans_idx]
            # Replace the blank ("_") with the correct choice
            filled_sentence = sentence_with_blank.replace("_", correct_choice)
            prompt_str += filled_sentence + "\n\n"
        else:
            # If no gold label is present, just show the sentence with blank
            prompt_str += sentence_with_blank + "\n\n"
    
    return prompt_str

def get_chunks(_iter, n:int=1) -> Iterator:

    _iter = tuple(_iter)
    for i, x in enumerate(_iter):
        yield x

def get_batched(batches, n: int=1) -> Iterator:
    batch = get_chunks(batches, n=n)
    yield from batch