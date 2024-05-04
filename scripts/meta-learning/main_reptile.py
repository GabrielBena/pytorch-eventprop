# Meta-Training SNNs using MAML

import matplotlib.pyplot as plt
import numpy as np

import torch
from collections import OrderedDict

import random
from tqdm.notebook import tqdm
import pandas as pd
import seaborn as sns
import argparse
from torchviz import make_dot
from tqdm.notebook import tqdm, trange
from tqdm.notebook import trange

from yingyang.meta_dataset import get_all_datasets
from torchmeta.transforms import ClassSplitter

from eventprop.config import get_flat_dict_from_nested
from eventprop.training import encode_data
from eventprop.models import SNN

import wandb

## Data and Config

if __name__ == "__main__":

    data_config = {
        "seed": 42,
        "dataset": "ying_yang",
        "deterministic": True,
        "meta_batch_size": 10,
        "encoding": "latency",
        "T": 50,
        "dt": 1e-3,
        "t_min": 0,
        "t_max": 2,
        "data_folder": "../../../data/",
        "n_samples_per_task": 100,  # adaptation steps
        "n_tasks_per_split_train": 20,  # number of rotations
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
            "scale_0_sigma": 3.5,
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

    optim_config = {
        "meta-lr": 1e-2,
        "inner-lr": 1e-2,
        "optimizer": "adam",
        "gamma": 0.95,
    }

    meta_config = {
        "n_epochs": 20,
        "num_shots": 10,
        "n_samples_test": 1000,
        "first_order": True,
        "learn_step_size": False,
    }

    default_config = {
        "data": data_config,
        "model": model_config,
        "optim": optim_config,
        "meta": meta_config,
        "loss": loss_config,
    }
    config = get_flat_dict_from_nested(default_config)
    args = argparse.Namespace(**config)
    dims = [n_ins[config["dataset"]]]
    if config["n_hid"] is not None and isinstance(config["n_hid"], list):
        dims.extend(config["n_hid"])
    elif isinstance(config["n_hid"], int):
        dims.append(config["n_hid"])
    dims.append(n_outs[config["dataset"]])

    model = SNN(dims, **config).to(config["device"])
    init_params = OrderedDict(model.meta_named_parameters()).copy()
    ## REPTILE
    from eventprop.training import REPTILE

    reptile_trainer = REPTILE(model, default_config)
    train_accs = {
        "pre": [],
        "post": [],
    }

    test_accs = {
        "pre": [],
        "post": [],
    }

    all_accs = {
        "train": train_accs,
        "test": test_accs,
    }

    use_wandb = True
    if use_wandb:
        wandb.init(project="ying_yang_reptile", config=config)

    for ep in trange(100, position=0, desc="Epochs"):

        for trial, accs in all_accs.items():
            for acc in accs.values():
                acc.append([])

        for training_batch in meta_train_dataloader:
            outer_loss, results = reptile_trainer.get_outer_loss(
                training_batch, use_tqdm=True, train=True, position=1
            )
            train_accs["pre"][-1].append([results["meta_accs"]["pre"]])
            train_accs["post"][-1].append([results["meta_accs"]["post"]])

        for testing_batch in meta_test_dataloader:
            outer_loss, results = reptile_trainer.get_outer_loss(
                testing_batch, use_tqdm=False, train=False
            )
            test_accs["pre"][-1].append([results["meta_accs"]["pre"]])
            test_accs["post"][-1].append([results["meta_accs"]["post"]])

        if use_wandb:
            wandb.log(
                {
                    "pre_train_accs": train_accs["pre"][-1],
                    "post_train_accs": train_accs["post"][-1],
                    "pre_test_accs": test_accs["pre"][-1],
                    "post_test_accs": test_accs["post"][-1],
                }
            )
