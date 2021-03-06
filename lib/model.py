import torch
import torch.nn as nn
from .utils import LockedDropout

class REGMAPR(nn.Module):

    def __init__(self,
                 vocab_size,
                 embed_dim,
                 locked_dropout,
                 lstm_dim,
                 recurrent_dropout,
                 hidden_dim,
                 dropout,
                 classes):

        super(REGMAPR, self).__init__()

        self.classes = classes
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lockdrop = LockedDropout(locked_dropout)
        self.bilstm = nn.LSTM(
            embed_dim + 2, lstm_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
            dropout=recurrent_dropout
        )

        in_dim = lstm_dim * (4 if self.classes == 1 else 8)
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, classes)
        )


    @staticmethod
    def get_ma(sent, twin):

        ma = []
        for i, seq in enumerate(sent):
            ma.append([word in twin[i] for word in seq])

        return ma


    @staticmethod
    def get_pa(sent, twin):

        pa = []
        for i, seq in enumerate(sent):
            pa.append([0 for word in seq])

        return pa


    def forward_once(self, sent, twin):

        # Embedding + MA + PR
        ma = torch.tensor(self.get_ma(sent, twin), dtype=torch.float32)
        pa = torch.tensor(self.get_pa(sent, twin), dtype=torch.float32)
        embed = self.embed(sent)

        cat = [ma.unsqueeze(2), embed, pa.unsqueeze(2)]
        enc = torch.cat(cat, dim=2)

        # Locked dropout
        drop = self.lockdrop(enc)

        # BiLSTM + Max Pooling
        lstm = self.bilstm(drop)[0]
        out = torch.max(lstm, 1)[0]

        return out


    def forward(self, s1, s2):
        """REGMAPR forward

        Args:
            s1: Tensor shape [pad, batch]
            s2: Tensor shape [pad, batch]
        """

        s1, s2 = s1.t(), s2.t()

        # Siamese passes
        h1 = self.forward_once(s1, s2)
        h2 = self.forward_once(s2, s1)

        # Concatenate encodings
        if self.classes == 1:
            h12 = torch.cat([h1 * h2, torch.abs(h1 - h2)], dim=1)
        else:
            h12 = torch.cat([h1, h1 * h2, torch.abs(h1 - h2), h2], dim=1)

        # Hidden and scoring layers
        clf = self.fc(h12)

        # Exponential, clamp, for relatedness tasks
        if self.classes == 1:
            clf = clf.exp().clamp(0, 1)

        return clf
