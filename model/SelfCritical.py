import torch


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
    # TODO implement method to calculate Levenshtein distance
    pass


class SelfCritical(torch.autograd.Function):
    # TODO complete methods
    def __init__(self):
        pass

    def forward(self, loss, reward):
        pass

    def backward(self):
        pass
