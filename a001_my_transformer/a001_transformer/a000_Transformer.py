import math

import torch
from torch import nn
from a001_my_transformer.a001_transformer.a001_EncoderBlock import EncoderBlock
from a001_my_transformer.a001_transformer.a002_DecoderBlock import DecoderBlock


class MyTransformer(nn.Module):
    def __init__(self,
                 n_encoder_blocks,
                 n_decoder_blocks,
                 n_in_out_channels,
                 n_hidden_every_head_channels,
                 n_hidden_ffn_channels,
                 n_heads,
                 score_func,
                 dropout_ratio):
        super().__init__()
        self.encoder_block_list = nn.ModuleList(
            [EncoderBlock(
                n_in_out_channels=n_in_out_channels,
                n_hidden_every_head_channels=n_hidden_every_head_channels,
                n_hidden_ffn_channels=n_hidden_ffn_channels,
                n_heads=n_heads,
                score_func=score_func,
                dropout_ratio=dropout_ratio
            ) for _ in range(n_encoder_blocks)]
        )
        self.decoder_block_list = nn.ModuleList(
            [DecoderBlock(
                n_in_out_channels=n_in_out_channels,
                n_hidden_every_head_channels=n_hidden_every_head_channels,
                n_hidden_ffn_channels=n_hidden_ffn_channels,
                n_heads=n_heads,
                score_func=score_func,
                dropout_ratio=dropout_ratio
            ) for _ in range(n_decoder_blocks)]
        )

    def forward(self,
                encoder_input,
                decoder_input,
                valid_len_for_encoder,
                valid_len_for_decoder):
        """
        :param encoder_input: shape (batch_size, n_queries, n_embedding_dims)
        :param decoder_input: shape (batch_size, n_queries_of_target_sentence, n_embedding_dims)
        :param valid_len_for_encoder: shape (batch_size), representing valid length for each sentence
        :param valid_len_for_decoder: shape (batch_size), same meaning.
        n_queries might also be called as sequence length.
        n_embedding_dims might also be called as n_in_channels.
        """
        encoder_input = scaled_positional_encoding(encoder_input)

        encoder_output = encoder_input
        for encoder_block in self.encoder_block_list:
            encoder_output = encoder_block.forward(x=encoder_output,
                                                   will_be_keys=encoder_output,
                                                   will_be_values=encoder_output,
                                                   valid_len=valid_len_for_encoder)

        decoder_output = decoder_input
        for decoder_block in self.decoder_block_list:
            decoder_output = decoder_block.forward(x=decoder_output,
                                                   valid_len_from_decoder=valid_len_for_decoder,
                                                   will_be_keys_from_encoder=encoder_output,
                                                   will_be_values_from_encoder=encoder_output,
                                                   valid_len_from_encoder=valid_len_for_encoder)
        return decoder_output


def scaled_positional_encoding(x):
    """
    x shape (batch, n_queries, n_channels)
    n_queries: number of queries
    n_channels: also called the num of features or dim of embedding. Must be an even number.
    """
    n_channels = x.shape[-1]
    if n_channels % 2 != 0:
        raise ValueError("n_channels (embedding dims) of input must be an even number.")
    return x * math.sqrt(n_channels) + get_positional_encoding(x_shape=x.shape)


def get_positional_encoding(x_shape):
    """Get position encoding"""
    batch_size, n_queries, n_channels = x_shape
    row_indices = torch.arange(n_queries, dtype=torch.float32).reshape(n_queries, 1)
    column_indices_times_2 = torch.arange(start=0,
                                          end=n_channels,
                                          step=2,
                                          dtype=torch.float32).unsqueeze(0)
    column_indices_times_2 = torch.pow(10000, column_indices_times_2 / n_channels)
    temp = row_indices * column_indices_times_2

    result = torch.zeros(size=(n_queries, n_channels))
    result[:, 0::2] = torch.sin(temp)
    result[:, 1::2] = torch.cos(temp)
    return result.unsqueeze(0).expand(x_shape)


def try_transformer_0():
    n_encoder_blocks = 6
    n_decoder_blocks = 6
    n_in_out_channels = 512
    n_hidden_every_head_channels = 64
    n_hidden_ffn_channels = 2048
    n_heads = 8
    score_func = "dot"
    dropout_ratio = 0.1

    batch = 4
    n_queries_encoder = 100
    n_queries_decoder = 100

    model = MyTransformer(n_encoder_blocks=n_encoder_blocks,
                          n_decoder_blocks=n_decoder_blocks,
                          n_in_out_channels=n_in_out_channels,
                          n_hidden_every_head_channels=n_hidden_every_head_channels,
                          n_hidden_ffn_channels=n_hidden_ffn_channels,
                          n_heads=n_heads,
                          score_func=score_func,
                          dropout_ratio=dropout_ratio)
    model.train()

    encoder_input = torch.rand(size=(batch, n_queries_encoder, n_in_out_channels))
    decoder_input = torch.rand(size=(batch, n_queries_decoder, n_in_out_channels))
    valid_len_for_encoder = torch.ones(size=(batch, ))
    valid_len_for_decoder = torch.ones(size=(batch, ))

    result = model(encoder_input=encoder_input,
                   decoder_input=decoder_input,
                   valid_len_for_encoder=valid_len_for_encoder,
                   valid_len_for_decoder=valid_len_for_decoder)
    print(result.shape)


if __name__ == '__main__':
    try_transformer_0()
