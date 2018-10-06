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
        self.bilstm = nn.Sequential(
            nn.LSTM(
                embed_dim, lstm_dim,
                num_layers=1,
                bidirectional=True,
                dropout=recurrent_dropout
            ),
            # TODO: Max pooling across LSTM outputs 

        )
        self.fc = nn.Sequential(
            nn.Linear(lstm_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, classes)
        )


    def forward_once(self, sent, twin):
      
        # Embedding + MA + PR
        ma = torch.tensor([t in twin for t in sent], dtype=torch.float32)
        # TODO: Paraphrase (PR) lookup indicators
        embed = self.embed(sent)
        enc = torch.cat([ma.t(), embed], dim=1)

        # Locked dropout
        drop = self.lockdrop(enc)

        # TODO: BiLSTM with max pooling on each intermediate state


    def forward(self, s1, s2):

        # Siamese passes
        h1 = self.forward_once(s1)
        h2 = self.forward_once(s2)

        # Concatenate encodings
        # TODO: Only use difference and product, for relatedness tasks
        h12 = torch.cat([h1, h1 * h2, torch.abs(h1 - h2), h2])

        # Hidden and scoring layers
        clf = self.fc(h12)

        # Exponential, clamp, for relatedness tasks
        if self.classes == 1:
            clf = fc.exp().clamp(0, 1)

        return clf
