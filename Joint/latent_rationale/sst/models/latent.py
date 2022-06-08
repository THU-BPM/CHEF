#!/usr/bin/env python

import torch
from torch import nn

from latent_rationale.common.util import get_z_stats
from latent_rationale.common.classifier import Classifier, CHEFClassifier
from latent_rationale.common.latent import \
    DependentLatentModel, IndependentLatentModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


__all__ = ["LatentRationaleModel", "LatentRationaleModelForCHEF"]


class LatentRationaleModel(nn.Module):
    """
    Latent Rationale
    Categorical output version (for SST)

    Consists of:

    p(y | x, z)     observation model / classifier
    p(z | x)        latent model

    """
    def __init__(self,
                 vocab:          object = None,
                 vocab_size:     int = 0,
                 emb_size:       int = 200,
                 hidden_size:    int = 200,
                 output_size:    int = 1,
                 dropout:        float = 0.1,
                 layer:          str = "lstm",
                 dependent_z:    bool = False,
                 z_rnn_size:     int = 30,
                 selection:      float = 1.0,
                 lasso:          float = 0.0,
                 lambda_init:    float = 1e-4,
                 lagrange_lr:    float = 0.01,
                 lagrange_alpha: float = 0.99,
                 ):

        super(LatentRationaleModel, self).__init__()

        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.vocab = vocab
        self.selection = selection
        self.lasso = lasso

        self.z_rnn_size = z_rnn_size
        self.dependent_z = dependent_z

        self.embed = embed = nn.Embedding(vocab_size, emb_size, padding_idx=1)

        self.classifier = Classifier(
            embed=embed, hidden_size=hidden_size, output_size=output_size,
            dropout=dropout, layer=layer, nonlinearity="softmax")
        # import ipdb; ipdb.set_trace()
        if self.dependent_z:
            self.latent_model = DependentLatentModel(
                embed=embed, hidden_size=hidden_size,
                dropout=dropout, layer=layer)
        else:
            self.latent_model = IndependentLatentModel(
                embed=embed, hidden_size=hidden_size,
                dropout=dropout, layer=layer)

        self.criterion = nn.NLLLoss(reduction='none')

        # lagrange
        self.lagrange_alpha = lagrange_alpha
        self.lagrange_lr = lagrange_lr
        self.register_buffer('lambda0', torch.full((1,), lambda_init))
        self.register_buffer('lambda1', torch.full((1,), lambda_init))
        self.register_buffer('c0_ma', torch.full((1,), 0.))  # moving average
        self.register_buffer('c1_ma', torch.full((1,), 0.))  # moving average

    @property
    def z(self):
        return self.latent_model.z

    @property
    def z_layer(self):
        return self.latent_model.z_layer

    @property
    def z_dists(self):
        return self.latent_model.z_dists

    def predict(self, logits, **kwargs):
        """
        Predict deterministically.
        :param x:
        :return: predictions, optional (dict with optional statistics)
        """
        assert not self.training, "should be in eval mode for prediction"
        return logits.argmax(-1)

    def forward(self, x):
        """
        Generate a sequence of zs with the Generator.
        Then predict with sentence x (zeroed out with z) using Encoder.

        :param x: [B, T] (that is, batch-major is assumed)
        :return:
        """
        mask = (x != 1)  # [B,T]
        z = self.latent_model(x, mask)
        y = self.classifier(x, mask, z)

        return y

    def get_loss(self, logits, targets, mask=None, **kwargs):

        optional = {}
        selection = self.selection
        lasso = self.lasso

        loss_vec = self.criterion(logits, targets)  # [B]

        # main MSE loss for p(y | x,z)
        ce = loss_vec.mean()        # [1]
        optional["ce"] = ce.item()  # [1]

        batch_size = mask.size(0)
        lengths = mask.sum(1).float()  # [B]

        # z = self.generator.z.squeeze()
        z_dists = self.latent_model.z_dists

        # pre-compute for regularizers: pdf(0.)
        if len(z_dists) == 1:
            pdf0 = z_dists[0].pdf(0.)
        else:
            pdf0 = []
            for t in range(len(z_dists)):
                pdf_t = z_dists[t].pdf(0.)
                pdf0.append(pdf_t)
            pdf0 = torch.stack(pdf0, dim=1)  # [B, T, 1]

        pdf0 = pdf0.squeeze(-1)
        pdf0 = torch.where(mask, pdf0, pdf0.new_zeros([1]))  # [B, T]

        # L0 regularizer
        pdf_nonzero = 1. - pdf0  # [B, T]
        pdf_nonzero = torch.where(mask, pdf_nonzero, pdf_nonzero.new_zeros([1]))

        l0 = pdf_nonzero.sum(1) / (lengths + 1e-9)  # [B]
        l0 = l0.sum() / batch_size

        # `l0` now has the expected selection rate for this mini-batch
        # we now follow the steps Algorithm 1 (page 7) of this paper:
        # https://arxiv.org/abs/1810.00597
        # to enforce the constraint that we want l0 to be not higher
        # than `selection` (the target sparsity rate)

        # lagrange dissatisfaction, batch average of the constraint
        c0_hat = (l0 - selection)

        # moving average of the constraint
        self.c0_ma = self.lagrange_alpha * self.c0_ma + \
            (1 - self.lagrange_alpha) * c0_hat.item()

        # compute smoothed constraint (equals moving average c0_ma)
        c0 = c0_hat + (self.c0_ma.detach() - c0_hat.detach())

        # update lambda
        self.lambda0 = self.lambda0 * torch.exp(self.lagrange_lr * c0.detach())

        with torch.no_grad():
            optional["cost0_l0"] = l0.item()
            optional["target0"] = selection
            optional["c0_hat"] = c0_hat.item()
            optional["c0"] = c0.item()  # same as moving average
            optional["lambda0"] = self.lambda0.item()
            optional["lagrangian0"] = (self.lambda0 * c0_hat).item()
            optional["a"] = z_dists[0].a.mean().item()
            optional["b"] = z_dists[0].b.mean().item()

        loss = ce + self.lambda0.detach() * c0

        if lasso > 0.:
            # fused lasso (coherence constraint)

            # cost z_t = 0, z_{t+1} = non-zero
            zt_zero = pdf0[:, :-1]
            ztp1_nonzero = pdf_nonzero[:, 1:]

            # cost z_t = non-zero, z_{t+1} = zero
            zt_nonzero = pdf_nonzero[:, :-1]
            ztp1_zero = pdf0[:, 1:]

            # number of transitions per sentence normalized by length
            lasso_cost = zt_zero * ztp1_nonzero + zt_nonzero * ztp1_zero
            lasso_cost = lasso_cost * mask.float()[:, :-1]
            lasso_cost = lasso_cost.sum(1) / (lengths + 1e-9)  # [B]
            lasso_cost = lasso_cost.sum() / batch_size

            # lagrange coherence dissatisfaction (batch average)
            target1 = lasso

            # lagrange dissatisfaction, batch average of the constraint
            c1_hat = (lasso_cost - target1)

            # update moving average
            self.c1_ma = self.lagrange_alpha * self.c1_ma + \
                (1 - self.lagrange_alpha) * c1_hat.detach()

            # compute smoothed constraint
            c1 = c1_hat + (self.c1_ma.detach() - c1_hat.detach())

            # update lambda
            self.lambda1 = self.lambda1 * torch.exp(
                self.lagrange_lr * c1.detach())

            with torch.no_grad():
                optional["cost1_lasso"] = lasso_cost.item()
                optional["target1"] = lasso
                optional["c1_hat"] = c1_hat.item()
                optional["c1"] = c1.item()  # same as moving average
                optional["lambda1"] = self.lambda1.item()
                optional["lagrangian1"] = (self.lambda1 * c1_hat).item()

            loss = loss + self.lambda1.detach() * c1

        # z statistics
        if self.training:
            num_0, num_c, num_1, total = get_z_stats(self.latent_model.z, mask)
            optional["p0"] = num_0 / float(total)
            optional["pc"] = num_c / float(total)
            optional["p1"] = num_1 / float(total)
            optional["selected"] = 1 - optional["p0"]

        return loss, optional

from latent_rationale.nn.kuma_gate import KumaGate
from latent_rationale.common.util import get_encoder
from transformers import (
    AutoTokenizer,
    AutoModel
)

class LatentRationaleModelForCHEF(nn.Module):
    """
    Latent Rationale
    Categorical output version (for SST)
    Consists of:
    p(y | x, z)     observation model / classifier
    p(z | x)        latent model
    """
    def __init__(self,
                 bert_type:      str = 'bert-base-chinese',
                 emb_size:       int = 200,
                 hidden_size:    int = 200,
                 output_size:    int = 1,
                 dropout:        float = 0.1,
                 layer:          str = "lstm",
                 dependent_z:    bool = False,
                 z_rnn_size:     int = 30,
                 selection:      float = 1.0,
                 lasso:          float = 0.0,
                 lambda_init:    float = 1e-4,
                 lagrange_lr:    float = 0.01,
                 lagrange_alpha: float = 0.99,
                 nlayer: int = 2,
                 ):

        super(LatentRationaleModelForCHEF, self).__init__()

        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.selection = selection
        self.lasso = lasso

        enc_size = 768
        self.enc_layer = get_encoder('lstm', enc_size, self.hidden_size)
        self.z_layer = KumaGate(self.hidden_size * 2)

        self.z_rnn_size = z_rnn_size
        self.dependent_z = dependent_z

        self.classifier = CHEFClassifier(
            hidden_size=hidden_size, output_size=output_size,
            dropout=dropout, layer=layer, nonlinearity="softmax", nlayer=nlayer)

        self.criterion = nn.NLLLoss(reduction='none')

        # lagrange
        self.lagrange_alpha = lagrange_alpha
        self.lagrange_lr = lagrange_lr
        self.register_buffer('lambda0', torch.full((1,), lambda_init))
        self.register_buffer('lambda1', torch.full((1,), lambda_init))
        self.register_buffer('c0_ma', torch.full((1,), 0.))  # moving average
        self.register_buffer('c1_ma', torch.full((1,), 0.))  # moving average
        
        self.bert = AutoModel.from_pretrained(bert_type)

    # @property
    # def z(self):
    #     return self.latent_model.z
    #
    # @property
    # def z_layer(self):
    #     return self.latent_model.z_layer
    #
    # @property
    # def z_dists(self):
    #     return self.latent_model.z_dists

    def predict(self, logits, **kwargs):
        """
        Predict deterministically.
        :param x:
        :return: predictions, optional (dict with optional statistics)
        """
        assert not self.training, "should be in eval mode for prediction"
        return logits.argmax(-1)

    def forward(self, batch):
        # 325 到 343 调维度
        claim_ids = batch[0]
        evidences_ids = batch[1][0].unsqueeze(0) # [batchsize, evidence num, 256]
        labels = batch[2]
        device = labels.device
        # import ipdb;
        # ipdb.set_trace()
        # claim_ids = claim_ids[0][torch.nonzero(claim_ids[0],  as_tuple = False)].reshape(1, -1)
        # mask = torch.zeros(claim_ids.shape[1]).to(claim_ids.device)
        # mask[torch.nonzero(claim_ids[0], as_tuple=False)] = 1
        # mask = mask.unsqueeze(0)
        # claims_ebds = self.bert(claim_ids)
        claims_ebd = self.bert(
            claim_ids
        )[0]

        # [batchsize, 1, 768]
        # claims_ebd = claims_ebd.unsqueeze(0).unsqueeze(0)
        evidences_ebd = []
        # batchsize is only one
        for i in range(len(evidences_ids[0][:32])):
            evidence_ebd = self.bert(
                evidences_ids[0][i:i+1]
            )[0][0][0]
            evidences_ebd.append(evidence_ebd)
        evidences_ebd = torch.stack(evidences_ebd)
        evidences_ebd = evidences_ebd.unsqueeze(0) # [batchsize, evidence num, 768]
        # 计算latent_model(evidences_ebd) 原来的z = self.latent_model(x, mask)

        mask = torch.ones(1, evidences_ebd.shape[1]).bool().to(device)
        lengths = torch.tensor([evidences_ebd.shape[1]]).to(device)
        h, _ = self.enc_layer(evidences_ebd, mask, lengths)
        # 350-368
        z_dist = self.z_layer(h)
        # import ipdb;
        # ipdb.set_trace()
        if self.training:
            if hasattr(z_dist, "rsample"):
                z = z_dist.rsample()  # use rsample() if it's there
            else:
                z = z_dist.sample()  # [B, M, 1]
        else:
            # deterministic strategy
            p0 = z_dist.pdf(h.new_zeros(()))
            p1 = z_dist.pdf(h.new_ones(()))
            pc = 1. - p0 - p1  # prob. of sampling a continuous value [B, M]
            z = torch.where(p0 > p1, h.new_zeros([1]),  h.new_ones([1]))
            z = torch.where((pc > p0) & (pc > p1), z_dist.mean(), z)  # [B, M, 1]
        # mask invalid positions
        z = z.squeeze(-1)
        #
        self.z = z  # [B, T]
        self.z_dists = [z_dist]
        # y = self.classifier(x, mask, z)
        # import ipdb;
        # ipdb.set_trace()
        y = self.classifier(claims_ebd, evidences_ebd, z)
        
        return y

    def get_loss(self, logits, targets, mask=None, **kwargs):

        optional = {}
        selection = self.selection
        lasso = self.lasso

        loss_vec = self.criterion(logits, targets)  # [B]

        # main MSE loss for p(y | x,z)
        ce = loss_vec.mean()        # [1]
        optional["ce"] = ce.item()  # [1]

        # batch_size = mask.size(0)
        # lengths = mask.sum(1).float()  # [B]
        batch_size = 1
        lengths = torch.tensor([32]).to(targets.device)

        # z = self.generator.z.squeeze()
        z_dists = self.z_dists

        # pre-compute for regularizers: pdf(0.)
        if len(z_dists) == 1:
            pdf0 = z_dists[0].pdf(0.)
        else:
            pdf0 = []
            for t in range(len(z_dists)):
                pdf_t = z_dists[t].pdf(0.)
                pdf0.append(pdf_t)
            pdf0 = torch.stack(pdf0, dim=1)  # [B, T, 1]

        pdf0 = pdf0.squeeze(-1)
        # pdf0 = torch.where(mask, pdf0, pdf0.new_zeros([1]))  # [B, T]

        # L0 regularizer
        pdf_nonzero = 1. - pdf0  # [B, T]
        # pdf_nonzero = torch.where(mask, pdf_nonzero, pdf_nonzero.new_zeros([1]))

        l0 = pdf_nonzero.sum(1) / (lengths + 1e-9)  # [B]
        l0 = l0.sum() / batch_size

        # `l0` now has the expected selection rate for this mini-batch
        # we now follow the steps Algorithm 1 (page 7) of this paper:
        # https://arxiv.org/abs/1810.00597
        # to enforce the constraint that we want l0 to be not higher
        # than `selection` (the target sparsity rate)

        # lagrange dissatisfaction, batch average of the constraint
        c0_hat = (l0 - selection)

        # moving average of the constraint
        self.c0_ma = self.lagrange_alpha * self.c0_ma + \
            (1 - self.lagrange_alpha) * c0_hat.item()

        # compute smoothed constraint (equals moving average c0_ma)
        c0 = c0_hat + (self.c0_ma.detach() - c0_hat.detach())

        # update lambda
        self.lambda0 = self.lambda0 * torch.exp(self.lagrange_lr * c0.detach())

        with torch.no_grad():
            optional["cost0_l0"] = l0.item()
            optional["target0"] = selection
            optional["c0_hat"] = c0_hat.item()
            optional["c0"] = c0.item()  # same as moving average
            optional["lambda0"] = self.lambda0.item()
            optional["lagrangian0"] = (self.lambda0 * c0_hat).item()
            optional["a"] = z_dists[0].a.mean().item()
            optional["b"] = z_dists[0].b.mean().item()

        loss = ce + self.lambda0.detach() * c0

        if lasso > 0.:
            # fused lasso (coherence constraint)

            # cost z_t = 0, z_{t+1} = non-zero
            zt_zero = pdf0[:, :-1]
            ztp1_nonzero = pdf_nonzero[:, 1:]

            # cost z_t = non-zero, z_{t+1} = zero
            zt_nonzero = pdf_nonzero[:, :-1]
            ztp1_zero = pdf0[:, 1:]

            # number of transitions per sentence normalized by length
            lasso_cost = zt_zero * ztp1_nonzero + zt_nonzero * ztp1_zero
            # lasso_cost = lasso_cost * mask.float()[:, :-1]
            lasso_cost = lasso_cost.sum(1) / (lengths + 1e-9)  # [B]
            lasso_cost = lasso_cost.sum() / batch_size

            # lagrange coherence dissatisfaction (batch average)
            target1 = lasso

            # lagrange dissatisfaction, batch average of the constraint
            c1_hat = (lasso_cost - target1)

            # update moving average
            self.c1_ma = self.lagrange_alpha * self.c1_ma + \
                (1 - self.lagrange_alpha) * c1_hat.detach()

            # compute smoothed constraint
            c1 = c1_hat + (self.c1_ma.detach() - c1_hat.detach())

            # update lambda
            self.lambda1 = self.lambda1 * torch.exp(
                self.lagrange_lr * c1.detach())

            with torch.no_grad():
                optional["cost1_lasso"] = lasso_cost.item()
                optional["target1"] = lasso
                optional["c1_hat"] = c1_hat.item()
                optional["c1"] = c1.item()  # same as moving average
                optional["lambda1"] = self.lambda1.item()
                optional["lagrangian1"] = (self.lambda1 * c1_hat).item()

            loss = loss + self.lambda1.detach() * c1

        # z statistics
        if self.training:
            num_0, num_c, num_1, total = get_z_stats(self.z, mask)
            optional["p0"] = num_0 / float(total)
            optional["pc"] = num_c / float(total)
            optional["p1"] = num_1 / float(total)
            optional["selected"] = 1 - optional["p0"]

        return loss, optional
