import torch
from torch import nn
import numpy as np

from latent_rationale.common.util import get_encoder
from latent_rationale.nn.Transformer_encoder import TransformerModel


class Classifier(nn.Module):
    """
    The Encoder takes an input text (and rationale z) and computes p(y|x,z)

    Supports a sigmoid on the final result (for regression)
    If not sigmoid, will assume cross-entropy loss (for classification)

    """

    def __init__(self,
                 embed:        nn.Embedding = None,
                 hidden_size:  int = 200,
                 output_size:  int = 1,
                 dropout:      float = 0.1,
                 layer:        str = "rcnn",
                 nonlinearity: str = "sigmoid"
                 ):

        super(Classifier, self).__init__()

        emb_size = embed.weight.shape[1]

        self.embed_layer = nn.Sequential(
            embed,
            nn.Dropout(p=dropout)
        )

        self.enc_layer = get_encoder(layer, emb_size, hidden_size)

        # self.enc_layer = TransformerModel(in_features, hidden_size)

        if hasattr(self.enc_layer, "cnn"):
            enc_size = self.enc_layer.cnn.out_channels
        else:
            enc_size = hidden_size * 2

        self.output_layer = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(emb_size, output_size),
            nn.Sigmoid() if nonlinearity == "sigmoid" else nn.LogSoftmax(dim=-1)
        )

        self.report_params()

    def report_params(self):
        # This has 1604 fewer params compared to the original, since only 1
        # aspect is trained, not all. The original code has 5 output classes,
        # instead of 1, and then only supervise 1 output class.
        count = 0
        for name, p in self.named_parameters():
            if p.requires_grad and "embed" not in name:
                count += np.prod(list(p.shape))
        print("{} #params: {}".format(self.__class__.__name__, count))

    def forward(self, x, mask, z=None):

        rnn_mask = mask
        emb = self.embed_layer(x)
        import ipdb; ipdb.set_trace()
        # apply z to main inputs
        if z is not None:
            z_mask = (mask.float() * z).unsqueeze(-1)  # [B, T, 1]
            rnn_mask = z_mask.squeeze(-1) > 0.  # z could be continuous
            emb = emb * z_mask

        # z is also used to control when the encoder layer is active
        lengths = mask.long().sum(1)

        # encode the sentence
        _, final = self.enc_layer(emb, rnn_mask, lengths)

        # predict sentiment from final state(s)
        y = self.output_layer(final)

        return y

class CHEFClassifier(nn.Module):
    """
    The Encoder takes an input text (and rationale z) and computes p(y|x,z)

    Supports a sigmoid on the final result (for regression)
    If not sigmoid, will assume cross-entropy loss (for classification)

    """

    def __init__(self,
                 enc_size:     int = 768,
                 hidden_size:  int = 200,
                 output_size:  int = 1,
                 dropout:      float = 0.1,
                 layer:        str = "rcnn",
                 nonlinearity: str = "sigmoid",
                 nlayer: int = 2
                 ):

        super(CHEFClassifier, self).__init__()
        # self.enc_layer = get_encoder(layer, enc_size, hidden_size)
        # import ipdb; ipdb.set_trace()
        self.enc_layer = TransformerModel(enc_size, nlayers=nlayer)
        self.output_layer = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid() if nonlinearity == "sigmoid" else nn.LogSoftmax(dim=-1)
        )

        # self.report_params()

    def report_params(self):
        # This has 1604 fewer params compared to the original, since only 1
        # aspect is trained, not all. The original code has 5 output classes,
        # instead of 1, and then only supervise 1 output class.
        count = 0
        for name, p in self.named_parameters():
            if p.requires_grad and "embed" not in name:
                count += np.prod(list(p.shape))
        print("{} #params: {}".format(self.__class__.__name__, count))

    def forward(self, claims_ebd, evidences_ebd, z=None):
        # import ipdb; ipdb.set_trace()
        mask = torch.ones(1, evidences_ebd.shape[1]).bool().to(claims_ebd.device)
        rnn_mask = mask
        # import ipdb; ipdb.set_trace()
        # apply z to main inputs
        if z is not None:
            z_mask = (mask.float() * z).unsqueeze(-1)  # [B, T, 1]
            rnn_mask = z_mask.squeeze(-1) > 0.  # z could be continuous
            evidences_ebd = evidences_ebd * z_mask

        # z is also used to control when the encoder layer is active
        # mask = torch.ones(1, evidences_ebd.shape[1]+1).bool().to(claims_ebd.device)
        # lengths = mask.long().sum(1)
        # encode the sentence
        # import ipdb; ipdb.set_trace()
        final = self.enc_layer(torch.cat((claims_ebd, evidences_ebd), dim=1).permute(1, 0, 2))
        # rnn_mask, lengths
        # _, claim_final = self.enc_layer(claims_ebd, torch.ones(1, claims_ebd.shape[1]), torch.tensor([1]).long())

        # predict sentiment from final state(s)
        # y = self.output_layer(torch.cat((claim_final, final), dim=1))
        y = self.output_layer(final[0])

        return y
