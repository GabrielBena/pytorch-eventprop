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
    loss_fn = SpikeCELoss(alpha=args.alpha, xi=args.xi, beta=args.beta)
    init_params = OrderedDict(model.meta_named_parameters()).copy()
    ## iMAML

    import torchopt
    from torchopt.diff.implicit import ImplicitMetaGradientModule
    from torchopt.visual import make_dot

    class InnerNet(
        ImplicitMetaGradientModule,
        linear_solve=torchopt.linear_solve.solve_normal_cg(maxiter=5, atol=0),
    ):
        def __init__(self, meta_net, loss_fn, n_inner_iter, reg_param, optim_params):
            super().__init__()
            self.loss_fn = loss_fn
            self.meta_net = meta_net
            self.net = torchopt.module_clone(meta_net, by="deepcopy", detach_buffers=True)
            self.n_inner_iter = n_inner_iter
            self.reg_param = reg_param
            self.reset_parameters()
            self.optim_params = optim_params

        def reset_parameters(self):
            with torch.no_grad():
                for p1, p2 in zip(self.parameters(), self.meta_parameters()):
                    p1.data.copy_(p2.data)
                    p1.detach_().requires_grad_()

        def forward(self, x):
            if len(x.shape) > 3:
                x = x.transpose(0, 1).squeeze()
            return self.net(x)

        def objective(self, x, y):
            # single sample processing
            out, _ = self(x)
            loss = self.loss_fn(out, y)[0]
            regularization_loss = 0
            for p1, p2 in zip(self.parameters(), self.meta_parameters()):
                regularization_loss += 0.5 * self.reg_param * torch.sum(torch.square(p1 - p2))
            return loss + regularization_loss

        def solve(self, inputs, targets):
            # Adaptation fn
            params = tuple(self.parameters())
            inner_optim = torchopt.Adam(params, lr=self.optim_params["lr"])
            with torch.enable_grad():
                # Temporarily enable gradient computation for conducting the optimization
                for x, y, _ in zip(inputs, targets, range(self.n_inner_iter)):
                    loss = self.objective(x, y)
                    inner_optim.zero_grad()
                    loss.backward(inputs=params)
                    inner_optim.step()

            return self

    inner_net = InnerNet(
        model,
        loss_fn,
        n_inner_iter=10,
        reg_param=0,
        optim_params=inner_optim_config,
    )

    training_batch = next(iter(meta_train_dataloader))
    meta_sample_train, meta_sample_test = training_batch["train"], training_batch["test"]
    meta_sample_train = [d[0] for d in meta_sample_train]
    meta_sample_test = [d[0] for d in meta_sample_test]
    optimal_inner_net = inner_net.solve(*meta_sample_train)

    out_spikes, _ = optimal_inner_net(meta_sample_test[0])
    test_loss, _, first_spikes = loss_fn(out_spikes, meta_sample_test[1])
    test_acc = (first_spikes.argmin(dim=-1) == meta_sample_test[1]).float().mean()
    test_loss.backward()
