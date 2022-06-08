import torch
import numpy as np
import math
import torch.nn as nn


def get_input_gradient_norm(loss, model):

    # Using create_graph=True ensures we add gradient calculation to the graph, so
    # that we can use higher order gradient
    #params_to_consider = []
    #for name, param in model.roberta.embeddings.named_parameters():
    #    if 'embeddings' in name:
    #        params_to_consider.append(param)
    #grad_params = torch.autograd.grad(loss, params_to_consider, create_graph=True)

    print(len(list(model.parameters())))
    grad_params = torch.autograd.grad(loss, model.parameters(), create_graph=True)

    grad_params_list = []
    for gp in grad_params:
        grad_params_list.append(gp.view(-1))

    # L2 Norm
    grad_norm = torch.norm(torch.cat(grad_params_list))

    #param = grad_params[0] For checking the zero grad elements
    #print(param.numel() - param.nonzero().size(0))

    return grad_norm
