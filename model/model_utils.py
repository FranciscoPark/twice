from collections.abc import Iterator
from typing import Iterable, List, Literal
import torch

def get_chunks(_iter: Iterable, n: int = 1) -> Iterator:
    _iter = tuple(_iter)
    for i in range(0, len(_iter), n):
        yield _iter[i:i + n]

def get_batched(dataset, n: int = 1) -> Iterator:
    yield from get_chunks(dataset, n=n)

def pad_and_concat(
    tensors: List[torch.Tensor],
    padding_side: Literal["right", "left"] = "right",
    max_length: int = 512
) -> torch.Tensor:
    """
    Method for padding a list of tensors given the maximum tensor
    length in the batch. Used for batching inputs and continuations in
    seq2seq models.
    """
    assert (
        padding_side == "left" or padding_side == "right"
    ), f"Unrecognized padding type: '{padding_side}' not 'left' or 'right'"

    
    for i, tensor in enumerate(tensors):
        if len(tensor.shape) == 2:
            tensor = tensor.squeeze(0)  # squeeze, in case passed [1, seq] size
        tensor_len = tensor.shape[0]
        if tensor_len < max_length:
            if padding_side == "right":
                # right-pad
                tensors[i] = torch.cat(
                    [
                        tensor,  # [seq]
                        torch.zeros(
                            max_length - tensor_len,
                            dtype=torch.long,
                            device=tensor.device,
                        ),  # [padding_length - seq]
                    ],
                    dim=0,
                ).unsqueeze(0)
            else:
                # left-pad
                tensors[i] = torch.cat(
                    [
                        torch.zeros(
                            max_length - tensor_len,
                            dtype=torch.long,
                            device=tensor.device,
                        ),  # [padding_length - seq]
                        tensor,  # [seq]
                    ],
                    dim=0,
                ).unsqueeze(0)
        else:
            tensors[i] = tensor.unsqueeze(0)

    return torch.cat(tensors, dim=0)

