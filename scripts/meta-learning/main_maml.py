# Meta-Training SNNs using MAML

import matplotlib.pyplot as plt
import numpy as np

import torch
from collections import OrderedDict

import random

from tqdm import tqdm

# from tqdm.notebook import tqdm as tqdm_n
import pandas as pd
import seaborn as sns
import argparse
from torchviz import make_dot

from yingyang.meta_dataset import get_all_datasets
from torchmeta.transforms import ClassSplitter

from eventprop.config import get_flat_dict_from_nested
from eventprop.training import encode_data
from eventprop.models import SNN, SpikeCELoss
import tracemalloc

import os, sys
from pathlib import Path
from pathlib import Path

path = Path(__file__).parent.absolute()
# sys.path.append(Path.joinpath(path, "../../../torchopt"))

import torchopt

import wandb

## Data and Config

if __name__ == "__main__":

    data_config = {
        "seed": 42,
        "dataset": "ying_yang",
        "deterministic": True,
        "meta_batch_size": 1,
        "encoding": "latency",
        "T": 30,
        "dt": 1e-3,
        "t_min": 0,
        "t_max": 2,
        "data_folder": "../../../data/",
        "n_samples_per_task": 100,  # adaptation steps
        "n_tasks_per_split_train": 50,  # number of rotations
        "n_tasks_per_split_test": 20,  # number of rotations
        "n_tasks_per_split_val": 20,  # number of rotations
        "dataset_size": 1000,  # testing size
    }

    data_args = argparse.Namespace(**data_config)
    torch.manual_seed(data_config["seed"])
    np.random.seed(data_config["seed"])
    random.seed(data_config["seed"])

    data_config["dataset"] = data_config["dataset"]
    if data_config["deterministic"]:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    ## Rotation Ying Yang data for Meta Learning
    encode_tranform = lambda s: (encode_data(s[0], data_args), s[1])
    dataset_split = lambda d: ClassSplitter(
        d,
        num_train_per_class=data_config["dataset_size"],
        num_test_per_class=data_config["dataset_size"],
        shuffle=False,
    )

    (
        (
            meta_train_dataset,
            meta_val_dataset,
            meta_test_dataset,
        ),
        (
            meta_train_dataloader,
            meta_val_dataloader,
            meta_test_dataloader,
        ),
    ) = get_all_datasets(data_config, dataset_split, encode_tranform)

    ## Models

    model_config = {
        "model_type": "eventprop",
        "snn": {
            "T": data_config["T"],
            "dt": data_config["dt"],
            "tau_m": 20e-3,
            "tau_s": 5e-3,
        },
        "weights": {
            "init_mode": "kaiming_both",
            "scale_0_mu": 5,
            "scale_0_sigma": 2.5,
            "scale_1_mu": 5,
            "scale_1_sigma": 0.5,
            "n_hid": 120,
            "resolve_silent": False,
            "dropout": 0.0,
        },
        # "device": (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")),
        "device": torch.device("cpu"),
    }

    n_ins = {"mnist": 784, "ying_yang": 5 if data_config["encoding"] == "latency" else 4}
    n_outs = {"mnist": 10, "ying_yang": 3}

    loss_config = {
        "loss": "ce_temporal",
        "alpha": 3e-3,
        "xi": 0.5,
        "beta": 6.4,
    }

    inner_optim_config = {
        "optimizer": "adam",
        "lr": 1e-2,
        "weight_decay": 0.0,
        "gamma": 0.95,
        "beta_1": 0.9,
        "beta_2": 0.99,
    }

    outer_optim_config = {"step_size": 3e-3, "annealing": "linear"}

    meta_config = {
        "n_epochs": 100,
        "num_shots": 100,
        "n_samples_test": 1000,
        "first_order": True,
        "learn_step_size": False,
    }

    default_config = {
        "data": data_config,
        "model": model_config,
        "inner_optim": inner_optim_config,
        "outer_optim": outer_optim_config,
        "meta": meta_config,
        "loss": loss_config,
    }

    flat_config = get_flat_dict_from_nested(default_config)

    dims = [n_ins[flat_config["dataset"]]]
    if flat_config["n_hid"] is not None and isinstance(flat_config["n_hid"], list):
        dims.extend(flat_config["n_hid"])
    elif isinstance(flat_config["n_hid"], int):
        dims.append(flat_config["n_hid"])
    dims.append(n_outs[flat_config["dataset"]])

    use_wandb = False
    use_best_sweep_params = True
    sweep_id = "804krio6"
    best_params_to_use = {"inner_optim", "model", "loss"}

    if use_best_sweep_params:
        api = wandb.Api()
        sweep_path = f"m2snn/eventprop/{sweep_id}"
        sweep = api.sweep(sweep_path)
        best_run = sweep.best_run()
        best_params = best_run.config

    if use_best_sweep_params and best_params_to_use is not None:
        best_params = {
            k: best_params[k]
            for k in get_flat_dict_from_nested({k: default_config[k] for k in best_params_to_use})
        }

    if "seed" in best_params:
        best_params.pop("seed")
    if "device" in best_params:
        best_params.pop("device")

    if use_wandb:
        wandb.init(project="ying_yang_reptile", config=flat_config)
        config = wandb.config
    else:
        config = flat_config

    print("Overriding config with:", best_params)
    config.update(best_params, allow_val_change=True)

    args = argparse.Namespace(**config)

    model = SNN(dims, **config).to(config["device"])
    loss_fn = SpikeCELoss(alpha=args.alpha, xi=args.xi, beta=args.beta)
    init_params = OrderedDict(model.meta_named_parameters()).copy()
    ## MAML

    import torch.optim as optim

    meta_opt = optim.Adam(model.parameters(), lr=1e-3)
    inner_opt = torchopt.MetaAdam(
        model,
        lr=inner_optim_config["lr"],
        betas=[inner_optim_config["beta_1"], inner_optim_config["beta_2"]],
    )

    def adapt(model, training_batch, n_inner_iter=1000):
        # Adaptation fn
        params = tuple(model.parameters())
        meta_opt.zero_grad()

        meta_sample_train, meta_sample_test = training_batch["train"], training_batch["test"]
        meta_sample_train = [d[0][:100] for d in meta_sample_train]
        meta_sample_test = [d[0][:100] for d in meta_sample_test]

        # Temporarily enable gradient computation for conducting the optimization
        for x, y, _ in zip(*meta_sample_train, range(n_inner_iter)):

            out, _ = model(x)
            loss, _, first_spikes = loss_fn(out, y)
            inner_opt.step(loss)
            acc = (first_spikes.argmin(-1) == y).float().mean()

        out_spikes, _ = model(meta_sample_test[0])
        loss, _, first_spikes = loss_fn(out_spikes, meta_sample_test[1])
        acc = (first_spikes.argmin(-1) == meta_sample_test[1]).float().mean()

        print(f"Post Acc is {acc}")

        return acc, loss

    training_batch = next(iter(meta_train_dataloader))

    net_state_dict = torchopt.extract_state_dict(model, by="reference", detach_buffers=True)
    optim_state_dict = torchopt.extract_state_dict(inner_opt, by="reference")

    acc, loss = adapt(model, training_batch)

    torchopt.recover_state_dict(model, net_state_dict)
    torchopt.recover_state_dict(inner_opt, optim_state_dict)

    loss.backward()
    for n, p in model.named_parameters():
        print(n, p.grad.any() if p.grad is not None else p.grad)
