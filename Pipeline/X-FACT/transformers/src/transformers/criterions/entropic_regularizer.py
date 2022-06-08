import torch
import numpy as np
import math
import torch.nn as nn

class EntropicRegularizer(nn.Module):

    def __init__(self, entropy_lambda, ignore_index=None):

        super().__init__()

        # Specifies a target value that is ignored and
        # does not contribute to the input gradient
        self.ignore_index = ignore_index

        self.entropy_lambda = entropy_lambda

        self.log_softmax = torch.nn.LogSoftmax(dim=-1)



    def forward(self, scores, target, mask, num_classes, entropy_lambda=None, reduce=True, return_crossentropy=False, return_kl_loss=False):

        lprobs = self.log_softmax(scores)
        #print(lprobs.shape)
        #print(target.shape)

        cross_ent = torch.nn.NLLLoss(reduction='none')

        loss = cross_ent(lprobs, target)

        kl_div = nn.KLDivLoss(reduction='none')

        uniform = (1/num_classes)*torch.ones(scores.shape, dtype=torch.float32).to(scores.device)

        kl_loss = torch.sum(kl_div(lprobs, uniform), dim=1)

        #print('KL Shape is ')
        #print(kl_loss.shape)

        if entropy_lambda is None:
            entropy_lambda = self.entropy_lambda

        if mask is None:
            if reduce:
                return loss.sum()
            else:
                return loss

        cross_loss = mask * loss
        kl = (1 - mask) * entropy_lambda * kl_loss

        final_loss = cross_loss + kl

        #print(final_loss.shape)

        if reduce:
            final_loss = final_loss.sum()
            cross_loss = cross_loss.sum()
            kl = kl.sum()

        if return_crossentropy and return_kl_loss:
            return final_loss, cross_loss, kl
        else:
            return final_loss

