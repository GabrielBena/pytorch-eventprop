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
from eventprop.models import SNN
from eventprop.meta import REPTILE
import tracemalloc

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
    init_params = OrderedDict(model.meta_named_parameters()).copy()
    ## REPTILE

    reptile_trainer = REPTILE(model, config)

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

    trace_mem = False

    if trace_mem:

        tracemalloc.start()

    n_batchs = len(meta_train_dataloader)
    ep_pbar = tqdm(range(meta_config["n_epochs"]), position=0, desc="Epochs", leave=False)
    for ep in ep_pbar:
        for trial, accs in all_accs.items():
            accs["pre"].append([])
            accs["post"].append([])

        pbar = tqdm(meta_train_dataloader, position=1, desc="Meta-Batches", leave=False)
        for training_batch in pbar:

            outer_loss, results = reptile_trainer.get_outer_loss(
                training_batch, use_tqdm=True, train=True, position=1
            )
            train_accs["pre"][-1].extend(results["meta_accs"]["pre"])
            train_accs["post"][-1].extend(results["meta_accs"]["post"])
            desc = (
                f"Train Acc: {np.mean(train_accs['pre'][-1])} -> {np.mean(train_accs['post'][-1])}"
            )

            pbar.set_description(desc)
            if use_wandb:
                wandb.log(
                    {
                        "pre_train_accs": train_accs["pre"][-1][-1],
                        "post_train_accs": train_accs["post"][-1][-1],
                    }
                )

        for testing_batch in meta_test_dataloader:
            outer_loss, results = reptile_trainer.get_outer_loss(
                testing_batch, use_tqdm=False, train=False
            )
            test_accs["pre"][-1].extend(results["meta_accs"]["pre"])
            test_accs["post"][-1].extend(results["meta_accs"]["post"])

            if use_wandb:
                wandb.log(
                    {
                        "pre_test_accs": test_accs["pre"][-1][-1],
                        "post_test_accs": test_accs["post"][-1][-1],
                    }
                )

        mean_test_accs = {
            "pre": np.mean(test_accs["pre"][-1]),
            "post": np.mean(test_accs["post"][-1]),
        }

        ep_pbar.set_description(
            f"Epochs: Mean Test Acc {mean_test_accs['pre']} -> {mean_test_accs['post']}"
        )

        torch.save(model.state_dict(), "reptile_model")

    if trace_mem:
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics("lineno")

        print("[ Top 10 ]")
        for stat in top_stats[:10]:
            print(stat)

        print("Done")
        tracemalloc.stop()
