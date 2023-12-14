import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 n_in_out_channels: int,
                 n_hidden_every_head_channels: int,
                 n_heads: int,
                 score_func: str,
                 have_look_ahead_mask: bool,
                 have_valid_len_mask: bool,
                 dropout_ratio: float):
        """
        :param n_in_out_channels: number of input channels. Also, should be out channels.
        :param n_hidden_every_head_channels: number of every head's q k v channels.
        in the original paper, is set to n_in_channels / n_heads.
        :param n_heads: number of multi-head attention heads
        :param score_func: attention score function to be used
        :param have_look_ahead_mask: whether to use mask attention
        :param have_valid_len_mask: whether to use valid length mask
        :param dropout_ratio: dropout ratio
        """
        super().__init__()
        # handle fields (properties) which following the constructor input parameters
        self.n_in_out_channels = n_in_out_channels
        self.n_hidden_every_head_channels = n_hidden_every_head_channels
        self.n_heads = n_heads

        if score_func == "dot":
            self.score_func = self.scaled_dot_product_score_func
        elif score_func == "additive":
            self.score_func = self.additive_score_func
        else:
            print(f"Not supported score function: {score_func}")
            exit()

        self.have_look_ahead_mask = have_look_ahead_mask
        self.have_valid_len_mask = have_valid_len_mask
        self.dropout_ratio = dropout_ratio

        # handle other fields
        self.w_q = nn.Linear(self.n_in_out_channels, self.n_hidden_every_head_channels * n_heads)
        self.w_k = nn.Linear(self.n_in_out_channels, self.n_hidden_every_head_channels * n_heads)
        self.w_v = nn.Linear(self.n_in_out_channels, self.n_hidden_every_head_channels * n_heads)
        self.aggregate_heads = nn.Linear(self.n_hidden_every_head_channels * n_heads, self.n_in_out_channels)
        self.dropout_layer = nn.Dropout(p=self.dropout_ratio)

    def forward(self, will_be_queries, will_be_keys, will_be_values, valid_len):
        """
        :param will_be_queries: shape (batch, n_queries, n_channels). Also called x.
        :param will_be_keys: same shape.
        :param will_be_values: same shape.
        :param valid_len: real sequence length, shape (batch_size)
        """
        # q, k, v shape after multiply w =(batch, n_queries, n_channels * n_heads)
        # then reshape to (batch, n_heads, n_queries, n_channels)
        q, k, v = (self.reshape_to_expose_heads(self.w_q(will_be_queries)),
                   self.reshape_to_expose_heads(self.w_k(will_be_keys)),
                   self.reshape_to_expose_heads(self.w_v(will_be_values)))
        attention_weights = torch.softmax(input=self.score_func(q, k, valid_len), dim=-1)
        # dropout on attention weights
        weight_sum_of_values = torch.matmul(self.dropout_layer(attention_weights), v)
        return self.aggregate_heads(self.reshape_back(weight_sum_of_values))

    def reshape_to_expose_heads(self, x: torch.Tensor):
        """
        x shape: (batch, n_queries, n_total_hidden_channels),
        where n_total_hidden_channels is n_heads * n_hidden_every_head_channels.
        reshape to (batch, n_queries, n_heads, n_hidden_every_head_channels)
        then permute to (batch, n_heads, n_queries, n_channels)
        """
        return (x.reshape(x.shape[0], x.shape[1], self.n_heads, self.n_hidden_every_head_channels)
                .permute(0, 2, 1, 3))

    def reshape_back(self, x: torch.Tensor):
        """
        x shape: (batch, n_heads, n_queries, n_hidden_every_head_channels)
        permute to (batch, n_queries, n_heads, n_hidden_every_head_channels)
        reshape to (batch, n_queries, n_total_hidden_channels)
        """
        # after permute, must assign to itself first to update x's info about shape.
        # because then x.shape is called. or the wrong shape before permute will be used.
        # mainly because permute() does not change memory inplace
        x = x.permute(0, 2, 1, 3)
        return x.reshape(x.shape[0], x.shape[1], self.n_hidden_every_head_channels * self.n_heads)

    def scaled_dot_product_score_func(self, q, k, valid_len: torch.Tensor):
        """
        q shape (batch, n_heads, n_queries, n_hidden_every_head_channels).
        k shape is the same.
        valid_len has 1 dim, length = batch_size
        """
        # according to the original paper, divided by sqrt(n_channels) is necessary to keep std=1
        # scores shape (batch, n_heads, n_queries, n_keys).
        # In self attention, n_queries = n_keys.
        scores = torch.matmul(q, k.transpose(-1, -2)) / torch.sqrt(torch.tensor(q.shape[-1]))
        return self.mask_operation(scores, valid_len)

    def additive_score_func(self, q, k, valid_len):
        # TODO
        return self.scaled_dot_product_score_func(q, k, valid_len)

    def mask_operation(self, scores, valid_len):
        """scores shape (batch, n_heads, n_queries, n_keys)."""
        batch_size, n_heads, n_queries, n_keys = scores.shape
        if self.have_look_ahead_mask:
            row_indices, col_indices = torch.triu_indices(row=n_queries,
                                                          col=n_keys,
                                                          offset=1)  # get 2 rows of the tensor respectively
            scores[:, :, row_indices, col_indices] = torch.finfo(scores.dtype).min
        if self.have_valid_len_mask:
            # expand valid_len view to (batch_size, n_heads, n_queries, n_keys)
            valid_len = (valid_len.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                         .expand(batch_size, n_heads, n_queries, n_keys))
            # expand indices view to (batch_size, n_heads, n_queries, n_keys)
            indices = (torch.arange(n_keys)
                       .unsqueeze(0).unsqueeze(0).unsqueeze(0)
                       .expand(batch_size, n_heads, n_queries, n_keys))
            mask_bool = (indices >= valid_len)
            scores[mask_bool] = torch.finfo(scores.dtype).min
        return scores

    def switch_mask_mode(self, have_look_ahead_mask, have_valid_len_mask):
        self.have_look_ahead_mask = have_look_ahead_mask
        self.have_valid_len_mask = have_valid_len_mask


def try_attention():
    batch_size, n_heads, n_queries, n_channels = 2, 8, 3, 4
    x = torch.ones(size=(batch_size, n_queries, n_channels))
    valid_len = torch.randint(low=1, high=2, size=(batch_size, ))
    mha = MultiHeadAttention(n_in_out_channels=n_channels,
                             n_hidden_every_head_channels=n_channels,
                             n_heads=n_heads,
                             score_func="dot",
                             have_look_ahead_mask=True,
                             have_valid_len_mask=True,
                             dropout_ratio=0)
    print(mha(x, x, x, valid_len))


if __name__ == '__main__':
    try_attention()
