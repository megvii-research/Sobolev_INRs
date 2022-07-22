import os

import kornia
import scipy.io.wavfile as wavfile

import torch


def get_data(data_root, filename, factor):

    rate, wav = wavfile.read(os.path.join(data_root, filename))
    print("Rate: %d" % rate)
    print("Raw data shape: ", wav.shape)

    wav = torch.tensor(wav).reshape(-1, 1)
    scale = torch.max(torch.abs(wav))
    wav = wav / scale  # (N, 1)

    grad = kornia.filters.spatial_gradient(
        wav.unsqueeze(0).unsqueeze(0), mode="diff", order=1, normalized=True
    ).squeeze()  # (2, N)
    grad = grad[1, :].reshape(-1, 1)  # (N, 1)

    coordinate = torch.linspace(0, len(wav) - 1, len(wav)).reshape(-1, 1)  # (N, 1)

    downsampled_wav = wav[::factor, :]
    downsampled_grad = grad[::factor, :]
    downsampled_coordinate = coordinate[::factor, :]

    return {
        "wav": wav,
        "grad": grad,
        "coordinate": coordinate,
        "downsampled_wav": downsampled_wav,
        "downsampled_grad": downsampled_grad,
        "downsampled_coordinate": downsampled_coordinate,
    }
