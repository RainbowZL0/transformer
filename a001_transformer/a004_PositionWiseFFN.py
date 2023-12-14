from torch import nn


class PositionWiseFFN(nn.Module):
    def __init__(self, n_in_out_channels, n_hidden_channels):
        """
        in and out channels are the same according to the paper.
        But n_hidden_channels can be changed. Paper has this set to 2048.
        """
        super().__init__()
        self.ffn_0 = nn.Linear(n_in_out_channels, n_hidden_channels)
        self.activation = nn.ELU()
        self.ffn_1 = nn.Linear(n_hidden_channels, n_in_out_channels)

    def forward(self, x):
        """
        :param x: shape (batch_size, n_queries, n_channels)
        """
        return self.ffn_1(self.activation(self.ffn_0(x)))
