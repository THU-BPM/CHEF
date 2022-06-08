import torch
import numpy as np
import math
import torch.nn as nn

class FocalLoss(nn.Module):

    def __init__(self, gamma, ignore_index=None):

        super().__init__()

        # Specifies a target value that is ignored and
        # does not contribute to the input gradient
        self.ignore_index = ignore_index

        self.gamma = gamma

        self.log_softmax = torch.nn.LogSoftmax(dim=-1)


    def forward(self, scores, target, gamma=None, return_nll_loss=False):
        """
        scores : unnormalized scores
        target : target labels
        gamma : if gamma is None, self.gamma is used
        """

        if gamma is None:
            gamma = self.gamma

        lprobs = self.log_softmax(scores)

        if target.dim() == lprobs.dim() - 1:
            target = target.unsqueeze(-1)


        probs = torch.exp(lprobs)

        correct_class_probs = torch.gather(probs, 1, target).squeeze(-1)

        crit = torch.nn.NLLLoss(reduction='none')

        weights = torch.pow((1 - correct_class_probs), gamma)

        if target.dim() == lprobs.dim():
            target = target.squeeze(-1)

        nll_loss = crit(lprobs, target)

        loss = weights * nll_loss

        loss = loss.sum()

        if return_nll_loss:
            return loss, nll_loss
        else:
            return loss



