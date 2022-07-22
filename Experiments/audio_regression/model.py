import math
from functools import partial

import torch
from torch import nn
import numpy as np

import diff_operators


class Sine(nn.Module):
    def __init__(self, w0):
        super().__init__()
        self.w0 = w0

    def forward(self, input):
        return torch.sin(self.w0 * input)


def init_weights_normal(m):
    if type(m) == nn.Linear:
        if hasattr(m, "weight"):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity="relu", mode="fan_in")


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, "weight"):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def sine_init(m, w0):
    with torch.no_grad():
        if hasattr(m, "weight"):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / w0, np.sqrt(6 / num_input) / w0)


class PosEncodingNeRF(nn.Module):
    def __init__(self, in_features, sidelength=None, fn_samples=None, use_nyquist=True):
        """__init__.

        :param in_features:
        :param sidelength: [3, height, width]
        :param fn_samples:
        :param use_nyquist:
        """
        super().__init__()

        self.in_features = in_features

        if self.in_features == 3:
            self.num_frequencies = 10
        elif self.in_features == 2:
            assert sidelength is not None
            if isinstance(sidelength, int):
                sidelength = (sidelength, sidelength)
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(
                    min(sidelength[0], sidelength[1])
                )
        elif self.in_features == 1:
            assert fn_samples is not None
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(fn_samples)

        self.out_dim = in_features + 2 * in_features * self.num_frequencies

    def get_num_frequencies_nyquist(self, samples):
        nyquist_rate = 1 / (2 * (2 * 1 / samples))
        return int(math.floor(math.log(nyquist_rate, 2)))

    def forward(self, coords):
        coords = coords.view(coords.shape[0], -1, self.in_features)

        coords_pos_enc = coords
        for i in range(self.num_frequencies):
            for j in range(self.in_features):
                c = coords[..., j]

                sin = torch.unsqueeze(torch.sin((2**i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2**i) * np.pi * c), -1)

                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)

        return coords_pos_enc.reshape(coords.shape[0], -1, self.out_dim)


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        w0,
        activations,
        out_features,
        hidden_features,
        num_hidden_layers,
        has_pos_encoding,
        length,
        fn_samples,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.w0 = w0
        self.activations = activations
        self.hidden_features = hidden_features
        self.num_hidden_layers = num_hidden_layers
        self.has_pos_encoding = has_pos_encoding
        self.length = length  # N, for normalizing input coordinate
        self.fn_samples = fn_samples  # N//factor, for positional encoding

        assert len(self.activations) == (self.num_hidden_layers + 1)

        activations_and_inits = {
            "sine": (
                Sine(self.w0),
                first_layer_sine_init,
                partial(sine_init, w0=self.w0),
            ),
            "relu": (nn.ReLU(inplace=True), init_weights_normal, init_weights_normal),
        }

        if self.has_pos_encoding:
            self.positional_encoding = PosEncodingNeRF(
                in_features=in_features, fn_samples=fn_samples, use_nyquist=True
            )
            in_features = self.positional_encoding.out_dim

        # network architecture
        net = []
        net.append(
            nn.Sequential(
                nn.Linear(in_features=in_features, out_features=hidden_features),
                activations_and_inits[self.activations[0]][0],
            )
        )  # input layer
        for i in range(num_hidden_layers):  # hidden layers
            net.append(
                nn.Sequential(
                    nn.Linear(
                        in_features=hidden_features, out_features=hidden_features
                    ),
                    activations_and_inits[self.activations[i + 1]][0],
                )
            )

        net.append(
            nn.Sequential(
                nn.Linear(in_features=hidden_features, out_features=out_features)
            )
        )  # output linear layer, without activation
        self.net = nn.Sequential(*net)

        self.net[0].apply(activations_and_inits[self.activations[0]][1])  # input layer
        for i in range(self.num_hidden_layers):  # hidden layers
            self.net[i + 1].apply(activations_and_inits[self.activations[i + 1]][2])
        self.net[-1].apply(
            activations_and_inits[self.activations[-1]][2]
        )  # output layer, initialize as hidden layers

        print("Network:\n", self)

    def normalize_coordinate(self, coordinate, length):
        normalized_coordinate = coordinate / (length - 1)  # [0 ~ N-1] to [0 ~ 1]
        normalized_coordinate -= 0.5  # [0 ~ 1] to [-0.5 ~ 0.5]
        normalized_coordinate *= 200.0  # [-0.5 ~ 0.5] to [-100, 100]

        return normalized_coordinate

    def forward(self, input_coordinate, mode="train"):
        """forward.

        :param input_coordinate: [(H//factor)*(W//factor), 1] for train mode, [H*W, 1] for test mode
        :param mode: 'train' or 'test'
        """

        if mode == "train":
            # Enables us to compute gradients w.r.t. coordinates
            original_coordinate = input_coordinate.clone().detach().requires_grad_(True)
            coordinate = self.normalize_coordinate(original_coordinate, self.length)
        elif mode == "test":
            coordinate = input_coordinate.clone()
            coordinate = self.normalize_coordinate(coordinate, self.length)

        if self.has_pos_encoding:
            coordinate = self.positional_encoding(coordinate).squeeze(1)

        pred = self.net(coordinate)  # (n, 1)

        if mode == "train":
            pred_grad = diff_operators.gradient(pred, original_coordinate)
            return {"pred": pred, "pred_grad": pred_grad}
        elif mode == "test":
            return {"pred": pred}
