from torch import nn


class AddNorm(nn.Module):
    def __init__(self, norm_length, dropout_ratio):
        """
        :param norm_length: the shape to be normalized as a unit. Use n_channels(embedding dims) to norm by word level.
        """
        super().__init__()
        self.dropout_layer = nn.Dropout(p=dropout_ratio)
        self.layer_norm = nn.LayerNorm(norm_length)

    def forward(self, x_original, x_after_operation):
        return self.layer_norm(x_original + self.dropout_layer(x_after_operation))
