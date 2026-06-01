from __future__ import annotations

import math


def splitmix64(value: int) -> int:
    mask = (1 << 64) - 1
    value = (int(value) + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    value = (value ^ (value >> 31)) & mask
    return int(value)


def choose_coprime_stride(length: int, seed: int) -> int:
    size = int(length)
    if size <= 1:
        return 1
    stride = (splitmix64(seed) % (size - 1)) + 1
    while math.gcd(stride, size) != 1:
        stride += 1
        if stride >= size:
            stride = 1
    return int(stride)


def compute_examples_per_positive(negative_sampler: object | None) -> float:
    if negative_sampler is None:
        return 1.0
    method = getattr(negative_sampler, "examples_per_positive", None)
    if callable(method):
        return max(1.0, float(method()))
    return 1.0


def compute_positive_batch_size(target_batch_size: int, negative_sampler: object | None) -> int:
    return max(1, int(target_batch_size) // int(max(1.0, compute_examples_per_positive(negative_sampler))))
