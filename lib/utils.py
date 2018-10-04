import torch
import torch.nn as nn

class LockedDropout(nn.Module):
    """Apply locked dropout

    Based on the dropout method described in Gal & Ghahramani (2016).
    Code sourced from: https://github.com/salesforce/awd-lstm-lm
    """

    def __init__(self, dropout=0.5):

        super().__init__()

        self.dropout = dropout

    
    def forward(self, x):

        if not self.training or not self.dropout:
        
            return x
        
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - self.dropout)
        mask = torch.tensor(m, requires_grad=False) / (1 - self.dropout)
        mask = mask.expand_as(x)
        
        return mask * x
