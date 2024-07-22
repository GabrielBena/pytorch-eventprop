import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

from snntorch.spikegen import rate, latency
import argparse
import numpy as np
import ast


def encode_data(dataset, args, labels=None):

    if isinstance(dataset, Dataset):
        data = dataset.data
        labels = dataset.targets
    else:
        data = dataset

    if not isinstance(data, torch.Tensor):
        data = torch.from_numpy(data)

    if isinstance(args, dict):
        args = argparse.Namespace(**args)

    if "latency" in args.encoding:
        if not "eventprop" in args.encoding:
            spike_data = latency(
                data,
                args.T if args.t_max is None else args.T - args.t_max,
                first_spike_time=args.t_min,
                normalize=True,
                linear=True,
                interpolate=False,
            ).flatten(start_dim=2)

            if args.t_max is not None:
                spike_data = torch.cat(
                    [spike_data, torch.zeros_like(spike_data[-args.t_max :])], 0
                )

        else:
            if args.t_max is None:
                args.t_max = int(args.T * 3 / 5)
            spike_data = args.t_min + (args.t_max - args.t_min) * (data < 0.5).view(
                data.shape[0], -1
            )
            spike_data = F.one_hot(spike_data.long(), int(args.T)).permute(2, 0, 1)

        if args.dataset == "ying_yang":
            t0_spike = torch.zeros_like(spike_data[..., 0])
            t0_spike[0] = 1.0
            spike_data = torch.cat([spike_data, t0_spike.unsqueeze(-1)], dim=-1)

    else:
        spike_data = rate(data, args.T, gain=0.7)
        if len(spike_data.shape) > 2:
            spike_data = spike_data.flatten(start_dim=2).float()

    if args.exclude_ambiguous:
        assert labels is not None, "Labels must be provided to exclude ambiguous data"

        (spike_data, labels), (mask, _) = exclude_ambiguous_data(spike_data, labels)
    else:
        mask = torch.full([len(data)], True)

    return (
        spike_data.transpose(0, 1),
        labels,
        mask,
    )  # n_samples x n_time_steps x n_features


def exclude_ambiguous_data(data, labels):

    data = data.transpose(0, 1)
    forbidden_coordinates = []
    sets = [
        set([str(s) for s in np.unique(data.argmax(1)[labels == i], axis=0).tolist()])
        for i in range(3)
    ]
    for i in range(3):
        for j in range(3):
            if i != j:
                intersection = sets[i].intersection(sets[j])
                for c in intersection:
                    if i != 2 and j != 2:
                        forbidden_coordinates.append(ast.literal_eval(c))
                    # print(f"Found forbidden coordinates: {ast.literal_eval(c)}")

    try:

        forbidden_indexs = torch.unique(
            torch.cat(
                [
                    torch.where((data.argmax(1) == f).all(-1))[0]
                    for f in torch.tensor(forbidden_coordinates)
                ]
            )
        )
    except RuntimeError:
        forbidden_indexs = torch.tensor([])

    mask = torch.full([len(data)], True)
    if len(forbidden_indexs):
        mask[forbidden_indexs] = False

    return (data[mask].transpose(0, 1), labels[mask]), (mask, forbidden_indexs)


def balance_dataset(data, labels):
    n_classes = len(np.unique(labels))
    n_samples = len(labels)
    n_samples_per_class = n_samples
    
