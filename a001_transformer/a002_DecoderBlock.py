from torch import nn

from a001_my_transformer.a001_transformer.a003_MultiHeadAttention import MultiHeadAttention
from a001_my_transformer.a001_transformer.a005_AddNorm import AddNorm
from a001_my_transformer.a001_transformer.a004_PositionWiseFFN import PositionWiseFFN


class DecoderBlock(nn.Module):
    def __init__(self,
                 n_in_out_channels: int,
                 n_hidden_every_head_channels: int,
                 n_hidden_ffn_channels: int,
                 n_heads: int,
                 score_func: str,
                 dropout_ratio: float):
        super().__init__()
        self.masked_multi_head_attention = MultiHeadAttention(n_in_out_channels=n_in_out_channels,
                                                              n_hidden_every_head_channels=n_hidden_every_head_channels,
                                                              n_heads=n_heads,
                                                              score_func=score_func,
                                                              have_look_ahead_mask=True,
                                                              have_valid_len_mask=True,
                                                              dropout_ratio=dropout_ratio)
        self.add_norm_0 = AddNorm(norm_length=n_in_out_channels,
                                  dropout_ratio=dropout_ratio)
        self.cross_multi_head_attention = MultiHeadAttention(n_in_out_channels=n_in_out_channels,
                                                             n_hidden_every_head_channels=n_hidden_every_head_channels,
                                                             n_heads=n_heads,
                                                             score_func=score_func,
                                                             have_look_ahead_mask=False,
                                                             have_valid_len_mask=True,
                                                             dropout_ratio=dropout_ratio)
        self.add_norm_1 = AddNorm(norm_length=n_in_out_channels,
                                  dropout_ratio=dropout_ratio)
        self.ffn = PositionWiseFFN(n_in_out_channels=n_in_out_channels,
                                   n_hidden_channels=n_hidden_ffn_channels)
        self.add_norm_2 = AddNorm(norm_length=n_in_out_channels,
                                  dropout_ratio=dropout_ratio)

    def forward(self,
                x,
                valid_len_from_decoder,
                will_be_keys_from_encoder,
                will_be_values_from_encoder,
                valid_len_from_encoder):
        if self.training:
            self.masked_multi_head_attention.switch_mask_mode(have_look_ahead_mask=True,
                                                              have_valid_len_mask=True)
        else:
            self.masked_multi_head_attention.switch_mask_mode(have_look_ahead_mask=True,
                                                              have_valid_len_mask=False)

        x = self.add_norm_0.forward(
            x_original=x,
            x_after_operation=self.masked_multi_head_attention.forward(will_be_queries=x,
                                                                       will_be_keys=x,
                                                                       will_be_values=x,
                                                                       valid_len=valid_len_from_decoder)
        )
        x = self.add_norm_1.forward(
            x_original=x,
            x_after_operation=self.cross_multi_head_attention.forward(will_be_queries=x,
                                                                      will_be_keys=will_be_keys_from_encoder,
                                                                      will_be_values=will_be_values_from_encoder,
                                                                      valid_len=valid_len_from_encoder))
        return self.add_norm_2.forward(
            x_original=x,
            x_after_operation=self.ffn.forward(x)
        )
