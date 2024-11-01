import random
from typing import List
from torch import Tensor
from augmentation import custom_augmentation


__all__ = [
    "input_int",
    "collate_fn"
]


def collate_fn(batch):
    if not batch[0][1]["supervised"] and len(batch) >= 2 and random.random() < 0.5:
        sample1 = batch.pop()
        sample2 = batch.pop()
        batch.append(custom_augmentation.mix_up(sample1, sample2))
    return tuple(zip(*batch))


def input_int(prompt, min_value: int = None, max_value: int = None):
    while True:
        content = input(prompt)
        if is_integer(content):
            value = int(content)
            if min_value is not None:
                if value < min_value:
                    prompt = f"number should not smaller than{min_value}，try again: "
                    continue
            if max_value is not None:
                if value > max_value:
                    prompt = f"number should not greater than{max_value}，try again: "
                    continue
            return value
        else:
            prompt = "invalid input, try again："


def is_integer(value):
    if value.startswith("-"):
        return value[1:].isdigit()
    else:
        return value.isdigit()


def sum_loss(x: List[Tensor]) -> Tensor:
    res = None
    for i in x:
        if res is None:
            res = i
        else:
            res = res + i
    return res

