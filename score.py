import os
from typing import Dict,List,Optional,Union,Tuple,Any
import torch

import pandas as pd

def score_single(sum_prob: List[List[float]], answer: List[int]) -> float:
    df = pd.DataFrame({'probs': sum_prob, 'answer': answer})
    
    # Compute predicted index (argmax) for each row
    df['pred'] = df['probs'].apply(lambda x: int(pd.Series(x).idxmax()))
    
    # Check correctness
    df['correct'] = df['pred'] == df['answer']
    
    # Accuracy
    acc = float(round(df['correct'].mean() * 100, 2))

    # Optional: print correct indices
    # correct_indices = df[df['correct']].index.tolist()
    # print("Correctly answered question indices:", correct_indices)

    return acc
# def score_single(sum_prob: List[List[float]], answer: List[int]) -> float:
    
#     probs = torch.tensor((sum_prob))         # shape: [N, num_choices]
#     preds = torch.argmax(probs, dim=1)       # predicted indices
#     answers = torch.tensor(answer)           # ground truth
#     correct = (preds == answers).sum().item()
#     acc = round((correct / len(answer)) * 100, 2)
#     return acc

def score_ll(sum_prob: List[List[float]], answer: List[int], length: List[List[int]]) -> Tuple[float, float]:
    model_answers = []
    token_norm_model_answers = []

    for log_likelihoods, lengths in zip(sum_prob, length):
        # Raw max log-likelihood
        model_answer = torch.tensor(log_likelihoods).argmax().item()
        model_answers.append(model_answer)

        # Length-normalized log-likelihoods
        token_norm = [ll / l for ll, l in zip(log_likelihoods, lengths)]
        token_norm_answer = torch.tensor(token_norm).argmax().item()
        token_norm_model_answers.append(token_norm_answer)

    correct = sum(pred == gt for pred, gt in zip(model_answers, answer))
    token_norm_correct = sum(pred == gt for pred, gt in zip(token_norm_model_answers, answer))
    total = len(answer)

    acc = round((correct / total) * 100, 2)
    token_acc_norm = round((token_norm_correct / total) * 100, 2)
    return acc, token_acc_norm

# def padding(sum_prob:  List[List[float]])-> List[List[float]]:
#     max_len = max(len(row) for row in sum_prob)
#     #print(max_len)
#     padded_sum_prob = [row + [0.0] * (max_len - len(row)) for row in sum_prob]
#     #print(len(padded_sum_prob),len(padded_sum_prob[0]))
#     return padded_sum_prob
    
#TODO shape unifi
def score(
    sum_prob: Any,  # shape: [num_question ,2]
    answer: List[int]
) -> Tuple[float, float]:
    """
    sum_prob: List of two [num_questions, num_choices] matrices:
        - sum_prob[0]: probs from uppercase candidates (A, B, ...)
        - sum_prob[1]: probs from lowercase candidates (a, b, ...)
    answer: List of correct answer indices
    """
    # Pad each row in sum_prob to the same length before unpacking
    first_column, second_column = map(list, zip(*sum_prob))
    acc_upper = score_single(first_column, answer)
    acc_lower = score_single(second_column, answer)
    return acc_upper, acc_lower


    
        


   
    