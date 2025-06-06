from transformers import PreTrainedTokenizer, PreTrainedModel
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from model.model_utils import *
from typing import List, Union
from tqdm import tqdm

def model_running(dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, batch_size: int = 1
                 ):
    
    if tokenizer.pad_token == None:
        tokenizer.pad_token = tokenizer.eos_token
    
    chunks = get_batched(dataset=dataset, n=batch_size)

    pbar = tqdm(total=len(dataset), desc="Running Single Selection Task")

    result = [0] * len(dataset)
    for chunk in chunks:
        inputs = []
        for input, _, input_tok, cand_list_tok, idx in chunk:
            inputs.append(input)

        
        with torch.no_grad():
            inputs_tok = tokenizer(inputs, return_tensors="pt", padding=True, add_special_tokens=True).to(model.device)
            # print(inputs_tok)
            outputs = model(**inputs_tok)
            logits = outputs.logits # (batch, seq_len, vocab_size)
            attention_mask = inputs_tok['attention_mask']  # (batch, seq_len)

            # Calculate the position of the last actual token in each sentence (excluding padding)
            last_token_indices = attention_mask.sum(dim=1) - 1  # (batch, )

            # Collect the logits of the last token for each sentence
            last_logits = logits[torch.arange(logits.size(0)), last_token_indices]  # (batch, vocab)

            probs = F.softmax(last_logits, dim=-1)

        
        
        for batch_idx, (input, cand_list, _, cand_list_tok, idx) in enumerate(chunk):
            large = []
            small = []
            for i in range(len(cand_list_tok[0])):
                large_token_id = cand_list_tok[0][i]
                small_token_id = cand_list_tok[1][i]
                large_prob = probs[batch_idx][large_token_id].item()
                small_prob = probs[batch_idx, small_token_id].item()
                # print(f"[{batch_idx}] Large: {large_prob:.6f}, Small: {small_prob:.6f}")

                large.append(large_prob)
                small.append(small_prob)
            
            result[idx] = [large, small]
            pbar.update(1)

        # batched_inputs = pad_and_concat(inputs, padding_side="right")

        # with torch.no_grad():
        #     outputs = model(input_ids = batched_inputs)
        #     logits = outputs.logits[:, -1, :]
        
        # probs = F.softmax(logits, dim=-1)
        # print(probs)

    return result