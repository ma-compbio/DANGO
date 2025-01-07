import numpy as np
import torch
from tqdm import tqdm, trange
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score, pairwise_distances
from concurrent.futures import as_completed, ProcessPoolExecutor
from pybloom_live import ScalableBloomFilter
from copy import deepcopy
from scipy.stats import pearsonr, spearmanr
import json
import string
import random
import torch.nn as nn
import torch.nn.functional as F


def build_adj_matrix(input, size):
    a = np.zeros((size, size))
    a[input[:, 0].astype('int'), input[:, 1].astype('int')] += input[:, 2]
    a += a.T
    # a /= (np.sum(np.abs(a), axis=-1, keepdims=True) + 1e-15)
    a = a[:, np.sum(a != 0, axis=0) > 0]
    a[np.isnan(a)] = 0.0
    return a


# Log cosh regression loss
def log_cosh(pred, truth, sample_weight=None):
    ey_t = truth - pred
    if sample_weight is not None:

        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)) * sample_weight)
    else:
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))


# Ranking loss
def ranking_loss(pred, truth):
    diff = (pred.view(-1, 1) - pred.view(1, -1)).view(-1)
    diff_w = (truth.view(-1, 1) - truth.view(1, -1)).view(-1)
    thres = 0
    mask_rank = torch.abs(diff_w) > thres
    diff = diff[mask_rank]
    diff_w = diff_w[mask_rank]
    label = (diff_w > 0).float()
    loss = F.binary_cross_entropy_with_logits(diff, label)

    return loss


def sparse_mse(y_pred, y_true, lambda_=1.0):
    pos = torch.mean(torch.sum((y_true.ne(0).type(torch.float) * (y_true - y_pred)) ** 2, dim=-1) / (
            torch.sum(y_true.ne(0).type(torch.float), dim=-1) + 1e-15))
    neg = torch.mean(torch.sum((y_true.eq(0).type(torch.float) * (y_true - y_pred)) ** 2, dim=-1) / (
            torch.sum(y_true.eq(0).type(torch.float), dim=-1) + 1e-15))
    return pos + neg * lambda_


def randomString(stringLength=8):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))


def get_config():
    c = open("./config.JSON", "r")
    return json.load(c)


def add_padding_idx(vec):
    if len(vec.shape) == 1:
        return np.asarray([np.sort(np.asarray(v) + 1).astype('int')
                           for v in tqdm(vec)])
    else:
        vec = np.asarray(vec) + 1
        vec = np.sort(vec, axis=-1)
        return vec.astype('int')


def np2tensor_hyper(vec, dtype):
    vec = np.asarray(vec)
    if len(vec.shape) == 1:
        return [torch.as_tensor(v, dtype=dtype) for v in vec]
    else:
        return torch.as_tensor(vec, dtype=dtype)


def pass_(x):
    return x


def roc_auc_cuda(y_true, y_pred, balance=False):
    try:
        y_true = y_true.cpu().detach().numpy().reshape((-1, 1))
        y_pred = y_pred.cpu().detach().numpy().reshape((-1, 1))
    except:
        y_true = y_true.reshape((-1, 1))
        y_pred = y_pred.reshape((-1, 1))

    if balance:
        pos = np.where(y_true == 1.0)[0]
        neg = np.where(y_true == 0.0)[0]
        num = int(min(len(pos), len(neg)))
        if num == 0:
            print(y_true)
        pos = np.random.permutation(pos)[:num]
        neg = np.random.permutation(neg)[:num]
        index = np.concatenate([pos, neg])
        y_true, y_pred = y_true[index], y_pred[index]

    score1, score2 = roc_auc_score(
        y_true, y_pred), average_precision_score(
        y_true, y_pred)
    return score1, score2


def correlation_cuda(y_true, y_pred):
    try:
        y_true = y_true.cpu().detach().numpy().reshape((-1, 1))
        y_pred = y_pred.cpu().detach().numpy().reshape((-1, 1))
    except:
        y_true = y_true.reshape((-1, 1))
        y_pred = y_pred.reshape((-1, 1))

    return pearsonr(y_true.reshape((-1)), y_pred.reshape((-1)))[0], \
           spearmanr(y_true.reshape((-1)), y_pred.reshape((-1)))[0]


def accuracy(output, target):
    pred = output >= 0.5
    truth = target >= 0.5
    acc = torch.sum(pred.eq(truth))
    acc = float(acc) * 1.0 / (truth.shape[0] * 1.0)
    return acc
