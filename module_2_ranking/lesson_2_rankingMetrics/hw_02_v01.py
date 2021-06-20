import math
from typing import List, Optional, Union

import torch

def num_swapped_pairs(ys_true: torch.Tensor, ys_pred: torch.Tensor) -> int:
    
    n_pairs = 0
    ys_true_sorted = torch.sort(ys_true, descending=True)
    
    for i in range(len(ys_true_sorted[0]) - 1):
        for j in range(i, len(ys_true_sorted[0])):
            if (ys_pred[ys_true_sorted[1][i]] < ys_pred[ys_true_sorted[1][j]]) & \
                (ys_true_sorted[0][i] != ys_true_sorted[0][j]):
                n_pairs += 1
    
    return n_pairs

def compute_gain(y_value: float, gain_scheme: str) -> float:
    if gain_scheme == 'exp2':
        return 2 ** y_value - 1.
    else:
        return y_value + 0.

def dcg(ys_true: torch.Tensor, ys_pred: torch.Tensor, gain_scheme: str) -> float:
    ys_pred_sorted = torch.sort(ys_pred, descending=True)
    log2_list = [math.log2(x) for x in range(2, len(ys_pred) + 2)]
    
    dcg_val = 0.
    for i in range(len(log2_list)):
        dcg_val += compute_gain(ys_true[ys_pred_sorted[1]][i].item(), \
                                gain_scheme=gain_scheme) / log2_list[i]
    
    return dcg_val

def ndcg(ys_true: torch.Tensor, ys_pred: torch.Tensor, gain_scheme: str = 'const') -> float:
    ndcg_val = dcg(ys_true=ys_true, ys_pred=ys_pred, gain_scheme=gain_scheme) / \
        dcg(ys_true=ys_true, ys_pred=ys_true, gain_scheme=gain_scheme)
    return ndcg_val

def precission_at_k(ys_true: torch.Tensor, ys_pred: torch.Tensor, k: int) -> float:
    k_max = k
    k = min(k, len(ys_true))
    ys_pred_sorted = torch.sort(ys_pred, descending=True)
    ys_true_sorted = ys_true[ys_pred_sorted[1]]

    ys_true_k = ys_true_sorted[:k]
    ys_pred_k = ys_pred_sorted[0][:k]
    
    if sum(ys_true).item() == 0:
        prec_k_val = -1
    else:

        prec_k_val = sum(ys_true_k).item() / min(sum(ys_true).item(), k)
    
    return prec_k_val

def reciprocal_rank(ys_true: torch.Tensor, ys_pred: torch.Tensor) -> float:
    ys_pred_sorted = torch.sort(ys_pred, descending=True)
    ys_true_sorted = ys_true[ys_pred_sorted[1]]
    
    if sum(ys_true_sorted.numpy() == 1) == 0:
        return 0
    else:
        ys_true_idx = ((ys_true_sorted == 1).nonzero(as_tuple=True)[0])
        return 1 / (ys_true_idx.item() + 1)

def p_found(ys_true: torch.Tensor, ys_pred: torch.Tensor, p_break: float = 0.15) -> float:
    ys_pred_sorted = torch.sort(ys_pred, descending=True)
    ys_true_sorted = ys_true[ys_pred_sorted[1]] # pRel
    
    pLook = torch.zeros(len(ys_true))
    pLook[0] = 1
    
    for i in range(1, len(ys_true)):
        pLook[i] = pLook[i - 1] * (1 - ys_true_sorted[i - 1]) * (1 - p_break)
    
    pFound = sum(pLook * ys_true_sorted).item()
    
    return pFound

def average_precision(ys_true: torch.Tensor, ys_pred: torch.Tensor) -> float:
    if sum(ys_true) == 0:
        return -1
    else:
        ys_pred_sorted = torch.sort(ys_pred, descending=True)
        ys_true_sorted = ys_true[ys_pred_sorted[1]]
        ys_true_sum = sum(ys_true).item()
        
        avg_prec = 0
        recall_k_prev = 0
        for i in range(len(ys_true)):
            recall_k = sum(ys_true_sorted[:(i + 1)]).item() / ys_true_sum
            prec_k = sum(ys_true_sorted[:(i + 1)]).item() / (i + 1)
            avg_prec_i = (recall_k - recall_k_prev) * prec_k
            recall_k_prev = recall_k
            
            avg_prec += avg_prec_i
            
        return avg_prec