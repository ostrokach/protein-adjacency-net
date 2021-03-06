"""
This is a basic ``seq+adj - conv - seq+adj`` network.

"""
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numba import njit, prange

import pagnn.models.dcn
from pagnn.datavargan import dataset_to_datavar
from pagnn.models.common import AdjacencyConv, SequenceConv, SequentialMod
from pagnn.utils import expand_adjacency_tensor, padding_amount, reshape_internal_dim

logger = logging.getLogger(__name__)


MODEL_DATA_PATH = Path(__file__).resolve(strict=True).parent.joinpath("model_data")


def sparse_sum(input):
    # Need to test if this is faster...
    mone = torch.ones(input.shape[0], 1, dtype=torch.float)
    input = (input @ mone).squeeze()
    return input


def cut_to_max_distance(adj, max_distance):
    indices = adj._indices()
    values = adj._values()
    mask = values <= max_distance
    new_indices = indices[:, mask]
    new_values = values[mask]
    new_adj = torch.sparse_coo_tensor(
        new_indices, new_values, size=adj.size(), dtype=adj.dtype, device=adj.device
    )
    return new_adj


@njit(parallel=True)
def gen_barcode(distances, bins):
    barcode = np.zeros((len(distances), len(bins)), dtype=np.int32)
    for i in prange(len(distances)):
        a = distances[i]
        for j in range(len(bins)):
            if a < bins[j]:
                barcode[i, j] = 1
                break
    return barcode


def normalize_seq_distances(aa_distances):
    aa_distances_log_mean = 3.556_787_581_510_490_3
    aa_distances_log_std = 1.706_582_276_341_166_9

    aa_distances_log = torch.where(
        aa_distances > 0,
        torch.log(aa_distances) + 1,
        torch.zeros(len(aa_distances), dtype=aa_distances.dtype),
    )

    aa_distances_corrected = (aa_distances_log - aa_distances_log_mean) / aa_distances_log_std
    return aa_distances_corrected


def normalize_cart_distances(cart_distances):
    cart_distances_mean = 6.993_689_202_887_396_5
    cart_distances_std = 3.528_368_101_492_991

    cart_distances_corrected = (cart_distances - cart_distances_mean) / cart_distances_std
    return cart_distances_corrected


def get_self_interactions(seq_length):
    self_interactions = torch.sparse_coo_tensor(
        torch.stack(
            [
                torch.arange(seq_length * 2, dtype=torch.long),
                torch.tensor(
                    [i for ii in [[i, i] for i in range(seq_length)] for i in ii], dtype=torch.long
                ),
            ]
        ),
        torch.ones(seq_length * 2, dtype=torch.float),
        size=(seq_length * 2, seq_length),
        dtype=torch.float,
    ).to_dense()
    return self_interactions


class DistanceNet(nn.Module):
    def __init__(self, input_size, hidden_layer_size, barcode_size):
        super().__init__()

        self.input_size = input_size
        self.barcode_size = barcode_size
        self.hidden_layer_size = hidden_layer_size

        self.linear1 = nn.Linear(self.input_size, self.hidden_layer_size)
        self.linear2 = nn.Linear(self.hidden_layer_size, self.barcode_size)

        # self.reset_parameters()

    def reset_parameters(self):
        stdv = np.sqrt(self.hidden_layer_size)
        self.linear1.weight.data.normal_(0, stdv)
        self.linear1.bias.data.normal_(0, stdv)

    def forward(self, x):
        x = self.linear1(x)
        x = F.leaky_relu(x)
        x = self.linear2(x)
        x = F.leaky_relu(x)
        return x


class PairwiseConv(nn.Module):
    takes_extra_args = True

    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        bias,
        normalize,
        add_counts,
        max_distance,
        wself=False,
        barcode_method=None,
        barcode_size=12,
    ):
        super().__init__()
        # Validity checks
        assert barcode_method in [
            None,
            "separate",
            "separate.pretrained",
            "combined",
            "combined.pretrained",
        ]
        # Layers
        if barcode_method is not None:
            assert barcode_size % 2 == 0
            in_channels += barcode_size // 2
            if "separate" in barcode_method:
                self.seq_barcode_model = DistanceNet(1, 32, barcode_size // 2)
                self.cart_barcode_model = DistanceNet(1, 32, barcode_size // 2)
                if "pretrained" in barcode_method:
                    logger.info("Loading pretrained separate barcode model...")
                    self.seq_barcode_model.load_state_dict(
                        torch.load(MODEL_DATA_PATH.joinpath("seq_barcode_model.state"))
                    )
                    self.cart_barcode_model.load_state_dict(
                        torch.load(MODEL_DATA_PATH.joinpath("cart_barcode_model.state"))
                    )
            elif "combined" in barcode_method:
                self.seq_cart_barcode_model = DistanceNet(2, 64, barcode_size)
                if "pretrained" in barcode_method:
                    logger.info("Loading pretrained combined barcode model...")
                    self.seq_cart_barcode_model.load_state_dict(
                        torch.load(MODEL_DATA_PATH.joinpath("seq_cart_barcode_model.state"))
                    )
            else:
                raise Exception
        elif wself:
            in_channels += int(wself)

        if add_counts:
            out_channels = out_channels - 1

        self.spatial_conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=bias
        )

        # Save parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.add_counts = add_counts
        self.wself = wself
        self.max_distance = max_distance
        self.barcode_method = barcode_method
        self.barcode_size = barcode_size

    def forward(self, x: torch.Tensor, adj: torch.sparse.FloatTensor):
        if self.max_distance:
            adj = cut_to_max_distance(adj, self.max_distance)

        if self.barcode_method is not None:
            x = self._conv_wbarcode(x, adj)
        elif self.wself:
            x = self._conv_wself(x, adj)
        else:
            x = self._conv(x, adj)

        adj_sum = adj.to_dense().sum(dim=0)
        if self.normalize:
            adj_sum_clamped = adj_sum.clone()
            adj_sum_clamped[adj_sum_clamped == 0] = 1
            x = x / adj_sum_clamped
        if self.add_counts:
            x = torch.cat([x, adj_sum.expand(x.size(0), 1, -1)], dim=1)
        return x

    def _conv(self, x, adj):
        adj_pw = expand_adjacency_tensor(adj).to_dense()
        x = x @ adj_pw.transpose(0, 1)
        x = self.spatial_conv(x)
        x = x @ adj_pw[::2, :]
        return x

    def _conv_wbarcode(self, x, adj):
        adj_pw = expand_adjacency_tensor(adj).to_dense()
        seq_length = x.size(2)
        self_interactions = get_self_interactions(seq_length)
        adj_pw_wself = torch.cat([self_interactions, adj_pw], 0)
        x = x @ adj_pw_wself.transpose(0, 1)

        # Seq distances
        seq_distances = torch.cat(
            [
                torch.tensor([0] * seq_length, dtype=torch.float64),
                (adj._indices()[0, :] - adj._indices()[1, :]).abs().to(torch.float64),
            ]
        )
        seq_distances_norm = normalize_seq_distances(seq_distances)
        seq_distances_tensor = seq_distances_norm.to(torch.float32).unsqueeze(1)

        # Cart distances
        cart_distances = torch.cat(
            [torch.tensor([0.0] * seq_length, dtype=torch.float32), adj._values().to(torch.float32)]
        )
        cart_distances_norm = normalize_cart_distances(cart_distances)
        cart_distances_tensor = cart_distances_norm.to(torch.float32).unsqueeze(1)

        # Barcode
        if "separate" in self.barcode_method:
            seq_barcode = self.seq_barcode_model(seq_distances_tensor)
            cart_barcode = self.cart_barcode_model(cart_distances_tensor)
            barcode = torch.cat([seq_barcode, cart_barcode], 1)
        elif "combined" in self.barcode_method:
            seq_conv_distances_tensor = torch.cat([seq_distances_tensor, cart_distances_tensor], 1)
            barcode = self.seq_cart_barcode_model(seq_conv_distances_tensor)
        else:
            raise Exception

        # Add barcode to pairwise interaction tensor
        barcode = barcode.reshape(-1, self.barcode_size // 2)
        barcode_expanded = barcode.transpose(0, 1).expand(x.size(0), 6, -1)
        x = torch.cat([barcode_expanded, x], 1)

        # Create pairwise contact map
        x = self.spatial_conv(x)
        reverse_map = adj_pw_wself[0::2, :] + adj_pw_wself[1::2, :]
        # Surce: https://stackoverflow.com/a/41164472/2063031
        x = x @ reverse_map
        # x = torch.sum(x[:, :, :, None] * reverse_map[None, :, :], dim=2)
        # x, _ = torch.max(x[:, :, :, None] * reverse_map[None, :, :], dim=2)
        # assert torch.isclose(x_out_1, x_out_2, rtol=1e-5, atol=1e-6).all().item()
        return x

    def _conv_wself(self, x, adj):
        adj_pw = expand_adjacency_tensor(adj).to_dense()
        seq_length = x.size(2)
        self_interactions = get_self_interactions(seq_length)
        adj_pw_wself = torch.cat([self_interactions, adj_pw], 0)
        x = x @ adj_pw_wself.transpose(0, 1)
        barcode = torch.tensor(
            [1, 0] * seq_length + [0, 1] * (adj_pw.size(0) // 2), dtype=torch.float
        )
        x = torch.cat([barcode.expand(x.size(0), 1, -1), x], 1)
        x = self.spatial_conv(x)
        # TODO: Replace this with maxpooling equivalent
        x = x @ adj_pw_wself[::2, :]
        return x


class PairwiseSeqConv(nn.Module):
    def __init__(
        self, in_features, out_features, passthrough_fraction, bias=None, *, normalize, add_counts
    ):
        super().__init__()
        self.takes_extra_args = True
        # Parameters
        self.in_features = in_features
        self.out_features = out_features
        self.passthrough_fraction = passthrough_fraction
        self.num_seq_features = int(out_features * passthrough_fraction)
        # Layers
        self.seq_conv = nn.Conv1d(
            in_features, self.num_seq_features, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.pairwise_conv = PairwiseConv(
            in_features,
            out_features - self.num_seq_features,
            normalize=normalize,
            add_counts=add_counts,
            bias=False,
        )
        if bias:
            self.bias = torch.empty(out_features, dtype=torch.float32, requires_grad=True)
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.out_features)
        # self.conv.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, seq, adj):
        x = seq
        x_seq = self.seq_conv(x)
        assert x_seq.size(1) == self.num_seq_features
        x_adj = self.pairwise_conv(x, adj)
        x = torch.cat([x_seq, x_adj], 1)
        assert x.size(1) == self.out_features
        if self.bias is not None:
            x = (x.transpose(1, 2) + self.bias).transpose(2, 1)
        return x


class GraphConv(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, normalize=True, bias=True):
        super().__init__()
        # Parameters
        self.normalize = normalize
        self.takes_extra_args = True
        # Layers
        self.conv = nn.Conv1d(
            in_features, out_features, kernel_size=1, stride=1, padding=0, bias=False
        )
        if bias:
            self.bias = torch.empty(out_features, dtype=torch.float32, requires_grad=True)
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.conv.weight.size(1))
        self.conv.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input: torch.Tensor, adj: torch.sparse.FloatTensor):
        support = self.conv(input)
        adj = adj.to_dense()
        if self.normalize:
            adj_sum = adj.sum(dim=0)
            adj_sum[adj_sum == 0] = 1
            adj = adj_sum.diag().inverse() @ adj
        output = support @ adj.transpose(0, 1)
        if self.bias is not None:
            output = (output.transpose(1, 2) + self.bias).transpose(2, 1)
        return output


class FinalLayer(nn.Module):
    def __init__(self, input_size, output_size, bias=True):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size, bias=bias)

    def forward(self, x):
        x = x.max(2)[0]
        x = self.linear(x)
        x = x.unsqueeze(-1)
        return x


class RepeatPad(nn.Module):
    def __init__(self, padding):
        super().__init__()
        self.padding = padding

    def forward(self, x):
        if self.padding == 0:
            return x
        else:
            x_pad = x[:, :, : self.padding]
            x = torch.cat([x, x_pad], 2)
            return x


class Custom(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        logger.info("Network name: '%s'", self.__class__.__name__)

        # *** Parameters ***
        self.n_layers = 3
        #: Number of adjacency convolutions (keep as low as possible)
        self.n_convs = 0
        self.input_size = 20
        self.hidden_size = 64
        self.bottleneck_size = 0
        self.kernel_size = 3
        self.stride = 2
        self.padding = 1
        self.bias = False
        self.dropout_probability = 0.5
        self.passthrough_fraction = 1 / 3
        self.max_pool_kernel_size = 64

        # *** Layers ***
        self._configure_single_pairwise()

    def forward(self, seq, adjs):
        return self._forward_single_pairwise(seq, adjs)

    # Network with a single pairwise layer
    # || PairwiseConv
    # || Conv1D
    # || Final
    def _configure_single_pairwise(self):
        # === Layer 1 ===
        input_size = self.input_size
        hidden_size = self.hidden_size
        output_size = hidden_size

        self.layer_1 = SequentialMod(
            PairwiseConv(
                input_size,
                output_size,
                normalize=False,
                add_counts=False,
                bias=False,
                wself=True,
                barcode_method="combined.pretrained",
                max_distance=3,
            ),
            nn.ReLU(),
        )

        # === Layer N ===
        input_size = output_size
        hidden_size = int(input_size * 2)
        output_size = 1

        # self.layer_n = nn.Sequential(
        #     nn.Conv1d(
        #         input_size,
        #         hidden_size,
        #         kernel_size=self.kernel_size,
        #         stride=self.stride,
        #         padding=self.padding,
        #         bias=True,
        #     ),
        #     FinalLayer(hidden_size, output_size, bias=True),
        # )
        self.layer_n = nn.Sequential(
            nn.Conv1d(
                input_size,
                hidden_size,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                bias=True,
            ),
            RepeatPad(self.max_pool_kernel_size - 1),
            nn.MaxPool1d(self.max_pool_kernel_size),
            nn.Conv1d(hidden_size, output_size, kernel_size=1, bias=True),
        )

    def _forward_single_pairwise(self, seq, adjs):
        x = seq
        # Layer 1
        x = self.layer_1(x, adjs[0][0])
        # Layer N
        x = self.layer_n(x)
        return x

    # === Test 13 ===
    # || PairwiseConv + Conv1d (full)
    # || Conv1D
    # || Final
    def _configure_test_13(self):
        # === Layer 1 ===
        input_size = self.input_size
        hidden_size = self.hidden_size
        output_size = hidden_size

        self.layer_1 = SequentialMod(
            PairwiseSeqConv(
                input_size,
                output_size,
                self.passthrough_fraction,
                normalize=True,
                add_counts=True,
                bias=False,
            ),
            nn.ReLU(),
        )

        # === Layer N ===
        input_size = output_size
        hidden_size = int(input_size * 2)
        output_size = 1

        self.layer_n = nn.Sequential(
            nn.Conv1d(
                input_size,
                hidden_size,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                bias=True,
            ),
            FinalLayer(hidden_size, output_size, bias=True),
        )

    def _forward_test_13(self, seq, adjs):
        x = seq
        # Layer 1
        x = self.layer_1(x, adjs[0][0])
        # Layer N
        x = self.layer_n(x)
        return x

    # Network with a signle graph-conv layer

    def _configure_single_graphconv(self):
        input_size = self.input_size
        hidden_size = self.hidden_size
        output_size = int(hidden_size * 2)

        self.layer_1 = SequentialMod(
            #
            GraphConv(input_size, hidden_size),
            nn.ReLU(inplace=True),
        )
        self.linear_n = nn.Sequential(
            nn.Conv1d(
                hidden_size,
                output_size,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            ),
            FinalLayer(output_size, 1, bias=True),
        )

    def _forward_single_graphconv(self, seq, adjs):
        x = seq
        x = self.layer_1(x, adjs[0][0])
        x = self.linear_n(x)
        return x

    # Network with two seq+adj layers.

    def _configure_two_layer_seqadj(self):
        # === Layer 1 ===
        input_size = self.input_size
        hidden_size = self.hidden_size
        output_size = hidden_size

        self.layer_1_pre = nn.Conv1d(input_size, hidden_size, kernel_size=1, stride=1, padding=0)
        num_seq_features = int(hidden_size * self.passthrough_fraction)
        self.layer_1_seq = nn.Sequential()
        self.layer_1_adj = SequentialMod(
            PairwiseConv(int(hidden_size - num_seq_features), int(hidden_size - num_seq_features))
        )
        self.layer_1_post = nn.Sequential(nn.ReLU(), nn.Dropout(p=self.dropout_probability))

        # nn.MaxPool1d(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

        # === Layer 2 ===
        input_size = output_size
        hidden_size = int(input_size * 2)
        output_size = hidden_size

        self.layer_2_pre = nn.Sequential(
            nn.Conv1d(
                input_size,
                output_size,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            )
        )
        num_seq_features = int(hidden_size * self.passthrough_fraction)
        self.layer_2_seq = nn.Sequential()
        self.layer_2_adj = SequentialMod(
            PairwiseConv(int(hidden_size - num_seq_features), int(hidden_size - num_seq_features))
        )
        self.layer_2_post = nn.Sequential(nn.ReLU())

        # === Layer N ===
        input_size = output_size
        output_size = 1

        self.layer_n = FinalLayer(input_size, output_size, bias=True)

    def _forward_two_layer_seqadj(self, seq, adjs):
        x = seq
        # Layer 1
        x = self.layer_1_pre(x)
        num_seq_features = int(x.size(1) * self.passthrough_fraction)
        x_seq = x[:, :num_seq_features, :]
        x_adj = x[:, num_seq_features:, :]
        x_seq = self.layer_1_seq(x_seq)
        x_adj = self.layer_1_adj(x_adj, adjs[0][0])
        x = torch.cat([x_seq, x_adj], 1)
        x = self.layer_1_post(x)
        # Layer 2
        x = self.layer_2_pre(x)
        num_seq_features = int(x.size(1) * self.passthrough_fraction)
        x_seq = x[:, :num_seq_features, :]
        x_adj = x[:, num_seq_features:, :]
        x_seq = self.layer_2_seq(x_seq)
        x_adj = self.layer_2_adj(x_adj, adjs[0][1])
        x = torch.cat([x_seq, x_adj], 1)
        x = self.layer_2_post(x)
        # Layer N
        x = self.layer_n(x)
        return x

    def _configure_encoder(self):
        conv_kwargs = dict(
            kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias
        )

        input_channels = self.input_size
        for i in range(0, self.n_layers):
            output_channels = int(input_channels * 2) if i > 0 else self.hidden_size
            negative_slope = 0.2 if i == 0 else 0.01

            # Input
            if i == 0:
                encoder_pre = nn.Sequential(
                    nn.Conv1d(
                        input_channels, output_channels // 2, kernel_size=1, stride=1, padding=0
                    )
                )
            elif i % 2 == 1:
                encoder_pre = nn.Sequential(
                    nn.Conv1d(input_channels, output_channels, **conv_kwargs)
                )
            else:
                encoder_pre = nn.Sequential()
            setattr(self, f"encoder_pre_{i}", encoder_pre)

            # Sequence Conv
            if i % 2 == 0:
                encoder_seq = SequentialMod(
                    SequenceConv(output_channels // 4, output_channels // 2, **conv_kwargs)
                )
            else:
                encoder_seq = SequentialMod()
            setattr(self, f"encoder_seq_{i}", encoder_seq)

            # Adjacency Conv
            if i % 2 == 0:
                encoder_adj = SequentialMod(
                    AdjacencyConv(output_channels // 4, output_channels // 2),
                    nn.Conv1d(output_channels // 2, output_channels // 2, **conv_kwargs),
                    # nn.LeakyReLU(negative_slope, inplace=True),
                    # nn.InstanceNorm1d(
                    #     output_channels // 4,
                    #     momentum=0.01,
                    #     affine=True,
                    #     track_running_stats=True,
                    # ),
                )
            else:
                encoder_adj = SequentialMod()
            setattr(self, f"encoder_adj_{i}", encoder_adj)

            # Output
            if i < (self.n_layers - 1):
                encoder_post = nn.Sequential(
                    nn.LeakyReLU(negative_slope, inplace=True),
                    nn.BatchNorm1d(
                        output_channels, momentum=0.01, affine=True, track_running_stats=True
                    ),
                )
            else:
                encoder_post = nn.Sequential()
            setattr(self, f"encoder_post_{i}", encoder_post)

            input_channels = output_channels

        logger.info("Final output_channels: %s", output_channels)

        if self.bottleneck_size == 0:
            self.linear_in = nn.Linear(output_channels, 1, bias=True)
            self.conv_in = nn.Conv1d(output_channels, 1, kernel_size=output_channels, bias=True)
        else:
            raise NotImplementedError
            self.linear_in = nn.Linear(2048, self.bottleneck_size, bias=True)
            self.conv_in = nn.Conv1d(
                512, self.bottleneck_size, kernel_size=4, stride=4, padding=0, bias=True
            )

        return input_channels

    def forward_bak(self, seq, adjs):
        x = seq

        for i in range(self.n_layers):
            x = getattr(self, f"encoder_pre_{i}")(x)
            x_seq = x[:, : x.shape[1] // 2, :]
            x_seq = getattr(self, f"encoder_seq_{i}")(x_seq, i, adjs)
            x_adj = x[:, x.shape[1] // 2 :, :]
            x_adj = getattr(self, f"encoder_adj_{i}")(x_adj, i, adjs)
            x = torch.cat([x_seq, x_adj], 1)
            x = getattr(self, f"encoder_post_{i}")(x)
            # logger.debug(f"Encoder layer: {i}, input shape: {x.shape}")

        if self.bottleneck_size == 0:
            x = x.max(2, keepdim=True)[0]
            x = self.linear_in(x.squeeze()).unsqueeze(-1)
        else:
            raise NotImplementedError
            pad_amount = padding_amount(x, 2048)  # 4 x 512
            if pad_amount:
                x = F.pad(x, (0, pad_amount))
            x = reshape_internal_dim(x, 1, 512)
            x = self.conv_in(x)

        return x

    def dataset_to_datavar(self, ds):
        return dataset_to_datavar(
            ds,
            n_convs=self.n_convs + 1,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            remove_diags=1 + self.kernel_size // 2,
            add_diags=0,
        )


pagnn.models.dcn.Custom = Custom
