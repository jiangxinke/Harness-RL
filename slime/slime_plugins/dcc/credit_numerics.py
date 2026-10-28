"""Numerical guards that preserve zero relative credit for identical scores."""
from collections import defaultdict

import torch


def zero_constant_groups(raw_scores, group_ids, advantages):
    """Keep upstream values except exact constant groups with at least two visits."""
    raw = torch.as_tensor(raw_scores).flatten()
    result = advantages.clone()
    if raw.numel() != result.numel() or raw.numel() != len(group_ids):
        raise ValueError("Normalization scores, groups and advantages must align")
    members = defaultdict(list)
    for index, group in enumerate(group_ids):
        members[group].append(index)
    corrections = []
    for indices in members.values():
        values = raw[indices]
        if len(indices) > 1 and bool(torch.all(values == values[0])):
            previous = result[indices].clone()
            result[indices] = 0
            if bool(torch.any(previous != 0)):
                corrections.append({"indices": indices, "constant_score": float(values[0]),
                                    "upstream_advantages": previous.tolist()})
    return result, corrections
