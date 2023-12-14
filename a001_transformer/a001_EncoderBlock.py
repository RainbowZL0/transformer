import torch
from torch import nn

from a001_my_transformer.a001_transformer.a003_MultiHeadAttention import MultiHeadAttention
from a001_my_transformer.a001_transformer.a005_AddNorm import AddNorm
from a001_my_transformer.a001_transformer.a004_PositionWiseFFN import PositionWiseFFN


class EncoderBlock(nn.Module):
    def __init__(self,
                 n_in_out_channels: int,
                 n_hidden_every_head_channels: int,
                 n_hidden_ffn_channels: int,
                 n_heads: int,
                 score_func: str,
                 dropout_ratio: float):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(n_in_out_channels=n_in_out_channels,
                                                       n_hidden_every_head_channels=n_hidden_every_head_channels,
                                                       n_heads=n_heads,
                                                       score_func=score_func,
                                                       have_look_ahead_mask=False,
                                                       have_valid_len_mask=True,
                                                       dropout_ratio=dropout_ratio)
        self.add_norm_0 = AddNorm(norm_length=n_in_out_channels,
                                  dropout_ratio=dropout_ratio)
        self.position_wise_ffn = PositionWiseFFN(n_in_out_channels=n_in_out_channels,
                                                 n_hidden_channels=n_hidden_ffn_channels)
        self.add_norm_1 = AddNorm(norm_length=n_in_out_channels,
                                  dropout_ratio=dropout_ratio)

    def forward(self,
                x: torch.Tensor,
                will_be_keys,
                will_be_values,
                valid_len):
        x = self.add_norm_0(x, self.multi_head_attention.forward(will_be_queries=x,
                                                                 will_be_keys=will_be_keys,
                                                                 will_be_values=will_be_values,
                                                                 valid_len=valid_len))
        return self.add_norm_1(x, self.position_wise_ffn.forward(x))


def try_encoder():
    x = torch.rand(size=(2, 3, 4))
    encoder = EncoderBlock(n_in_out_channels=4,
                           n_hidden_every_head_channels=4,
                           n_hidden_ffn_channels=4,
                           n_heads=8,
                           score_func="dot",
                           dropout_ratio=0)
    output = encoder(x=x,
                     will_be_keys=x,
                     will_be_values=x,
                     valid_len=torch.tensor([1, 2]))
    print(output)


if __name__ == '__main__':
    # get_position_encoding(2, 10)
    try_encoder()
