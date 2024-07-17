import torch.nn as nn
import torch.nn.functional as F


class FastText(nn.Module):
    def __init__(self, input_size, dim, count_classes, padding_idx=None):
        super().__init__()
        self.dim = dim
        if padding_idx is not None:
            input_size += 1

        self.A = nn.EmbeddingBag(input_size, dim, mode='mean', padding_idx=padding_idx)
        self.B = nn.Linear(self.dim, count_classes, bias=False)

        self.init_weights()

    def init_weights(self):
        nn.init.uniform_(self.A.weight, -1 / self.dim, 1 / self.dim)
        self.B.weight.data.zero_()

    def forward(self, x):
        embeds = self.A(x)
        # return self.B(embeds)
        return F.log_softmax(self.B(embeds), dim=-1)

