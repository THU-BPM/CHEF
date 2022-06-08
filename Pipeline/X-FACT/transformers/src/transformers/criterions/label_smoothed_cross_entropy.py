import torch
import numpy as np
import math
import torch.nn as nn

class LabelSmoothedCrossEntropy(nn.Module):

    def __init__(self, label_smoothing, ignore_index=None):

        super().__init__()
        assert 0.0 < label_smoothing <= 1.0

        # Specifies a target value that is ignored and
        # does not contribute to the input gradient
        self.ignore_index = ignore_index

        self.eps = label_smoothing
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, scores, target, reduce=True, return_nll_loss=False):
        """
        scores : unnormalized scores
        target : target labels
        """


        lprobs = self.log_softmax(scores)
        if target.dim() == lprobs.dim() - 1:
            target = target.unsqueeze(-1)

        nll_loss = -lprobs.gather(dim=-1, index=target)
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)



        if self.ignore_index is not None:
            pad_mask = target.eq(self.ignore_index)
            if pad_mask.any():
                nll_loss.masked_fill_(pad_mask, 0.)
                smooth_loss.masked_fill_(pad_mask, 0.)
        else:
            nll_loss = nll_loss.squeeze(-1)
            smooth_loss = smooth_loss.squeeze(-1)

        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        eps_i = self.eps / lprobs.size(-1)

        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss

        if return_nll_loss:
            return loss, nll_loss
        else:
            return loss
