from typing import Any

import torch
import Levenshtein


def calculate_reward(sample, predict, gt):
    """
                Calculate Levenshtein distance of sample sequence and predict sequence
                Args:
                    sample: sequence from sample
                    predict: sequence from prediction
                    gt: ground-truth sequence
                Returns:
                    reward: (dict) reward["sample"], reward["predict"]
                """
    with torch.no_grad:
        if sample.ndim == 1:
            sample.unsqueeze(0)
        if predict.ndim == 1:
            predict.unsqueeze(0)
        if gt.ndim == 1:
            predict.unsqueeze(0)
        batch_size = gt.shape[0]
        sample_reward = torch.zeros(batch_size)
        predict_reward = torch.zeros(batch_size)
        for i in range(batch_size):
            sample_reward[i] = Levenshtein.distance(sample[i], gt[i])
            predict_reward[i] = Levenshtein.distance(predict[i], gt[i])
        return {"sample": sample_reward, "predict": predict_reward}


class SelfCritical(torch.autograd.Function):
    # TODO complete methods
    def __init__(self):
        pass

    @staticmethod
    def forward(ctx: Any, tensor, reward):
        ctx.constant = reward
        return tensor * reward

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.constant, None
