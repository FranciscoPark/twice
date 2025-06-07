from transformers import AutoTokenizer, AutoModelForCausalLM

from loader.load import load
from model.huggingface import model_running
import torch
from score import *
from utils import *
from loader.task_config import MMLU_subject
import numpy as np
import logging # logging
from typing import Tuple, Any

def load_model_and_tokenizer(model_name: str) -> Tuple[Any, Any]:
    """Load the model and tokenizer."""
    
    token = os.environ.get("HUGGINGFACE_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto", 
        trust_remote_code=True
    )

    return model, tokenizer


def evaluate(args, tasks: List[str], language: str):
    model, tokenizer = load_model_and_tokenizer(args.model)

    logger = logging.getLogger(__name__) # logging # Get logger instance

    final_result = {}

    for task in tasks:

        final_result[task] = {}
        if task == "mmlu":
            mmlu_result = []

            for each in MMLU_subject:

                logger.info(f"Evaluating {each} in MMLU...") # logging 
                print(f"Evaluating {each} in MMLU...")


                # single without instruction
                dataset, answer = load(name=each, 
                                        instruction=False, 
                                        tokenizer=tokenizer,
                                        cand_type="large",
                                        num_shot=args.num_fewshot,
                                        fewshot_seed=args.seed,
                                        language=language,
                                        limit=args.limit)
                
                to_acc = model_running(dataset=dataset, 
                                    model=model, 
                                    tokenizer=tokenizer, 
                                    batch_size=args.batch_size)
                result = score(to_acc, answer)
                mmlu_result.append(result)
        

            mmlu_result = tuple(np.mean(mmlu_result, axis=0))
            final_result[task] = mmlu_result


        else:
            logger.info(f"Evaluating {task}...") # logging 
            print(f"Evaluating {task}...")
            
            # single without instruction
            dataset, answer, og_dataset = load(name=task, 
                                        instruction=False, 
                                        tokenizer=tokenizer,
                                        cand_type="large",
                                        num_shot=args.num_fewshot,
                                        fewshot_seed=args.seed,
                                        language=language,
                                        limit=args.limit)
            
            to_acc = model_running(dataset=dataset, 
                                model=model, 
                                tokenizer=tokenizer, 
                                batch_size=args.batch_size)
            result = score(to_acc, answer)
            print(f"{task}'s result: {result}")
                 
            final_result[task] = result


	
        logger.info(f"Completed evaluation for task: {task}") # logging
    
    save_results(final_result,args.model,seed= args.seed, shot=args.num_fewshot,tasks=tasks)
    print(final_result)
    
    logger.info(f"Complete, results saved") # logging

    return


if __name__ == "__main__":
    # model_name = "meta-llama/Llama-3.1-8B"
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    model, tokenizer = load_model_and_tokenizer(model_name)

    data_name = "hellaswag"
    dataset, answer, og_dataset = load(name=data_name, 
                            instruction=False, 
                            tokenizer=tokenizer,
                            cand_type="large",
                            num_shot=0,
                            fewshot_seed=1234,
                            language="Korean",
                            limit=10)
    
    for i in range(10):
        print("LOCALIZED: " , dataset[i][0])
        print("ORIGINAL: " , og_dataset[i][0])
        print("TRANSLATED: " , og_dataset[i][1])
        print("--------------------------------")
        
    to_acc = model_running(dataset=dataset, 
                        model=model, 
                        tokenizer=tokenizer, 
                        batch_size=16)
    result = score(to_acc, answer)
    print(f"{data_name}'s result: {result}")