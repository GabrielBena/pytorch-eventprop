import numpy as np
import wandb
import random
import torch
import torch.nn.functional as F
import argparse
from snntorch.functional.loss import (
    ce_temporal_loss,
    SpikeTime,
    ce_rate_loss,
    ce_count_loss,
)

from torchvision import datasets, transforms
from yingyang.dataset import YinYangDataset
from eventprop.models import SNN, SpikeCELoss, FirstSpikeTime
from eventprop.training import train_single_model
from eventprop.config import get_flat_dict_from_nested


def main(args, use_wandb=False):
    if isinstance(args, argparse.Namespace):
        config = vars(args)
    elif isinstance(args, dict):
        config = args
    else:
        raise ValueError(
            "Invalid type for run configuration, must be dict or Namespace"
        )

    config["device"] = config.get(
        "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    if use_wandb:
        if wandb.run is None:
            run = wandb.init(project="eventprop", entity="m2snn", config=config)
        else:
            run = wandb.run
        # Default values overwritten by potential sweep
        config = wandb.config

    # ------ Data ------
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])

    config["dataset"] = config["dataset"]
    if config["deterministic"]:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if config["dataset"] == "mnist":
        train_dataset = datasets.MNIST(
            config["data_folder"],
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        test_dataset = datasets.MNIST(
            config["data_folder"],
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )
    elif config["dataset"] == "ying_yang":
        train_dataset = YinYangDataset(size=60000, seed=config["seed"])
        test_dataset = YinYangDataset(size=10000, seed=config["seed"] + 2)

    else:
        raise ValueError("Invalid dataset name")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False, drop_last=True
    )

    # ------ Model ------

    n_ins = {"mnist": 784, "ying_yang": 5 if config["encoding"] == "latency" else 4}
    n_outs = {"mnist": 10, "ying_yang": 3}

    dims = [n_ins[config["dataset"]]]
    if config["n_hid"] is not None and isinstance(config["n_hid"], list):
        dims.extend(config["n_hid"])
    elif isinstance(config["n_hid"], int):
        dims.append(config["n_hid"])
    dims.append(n_outs[config["dataset"]])

    model = SNN(dims, **config).to(config["device"])
    if config.get("train_last_only", False):
        for n, layer in enumerate(model.layers):
            if n < len(model.layers) - 1:
                for p in layer.parameters():
                    p.requires_grad = False

    # ------ Training ------
    optimizers_type = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}
    optimizer = optimizers_type[config["optimizer"]](
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    if config.get("gamma", None) is not None:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=config["gamma"]
        )
    loaders = {"train": train_loader, "test": test_loader}

    if config["loss"] == "ce_temporal":
        if config["model_type"] == "snntorch":
            criterion = ce_temporal_loss()
        elif config["model_type"] == "eventprop":
            criterion = SpikeCELoss(config["xi"], config["tau_s"])
        else:
            raise ValueError("Invalid model type")
    elif config["loss"] == "ce_both":
        criterion = [ce_temporal_loss(), SpikeCELoss(config["xi"], config["tau_s"])]
    elif config["loss"] == "ce_rate":
        criterion = ce_rate_loss()
    elif config["loss"] == "ce_count":
        criterion = ce_count_loss()
    else:
        raise ValueError("Invalid loss type")

    # first_spike_fns = (SpikeTime().first_spike_fn, FirstSpikeTime.apply)
    first_spike_fns = SpikeTime.FirstSpike.apply

    args = argparse.Namespace(**config)
    train_results = train_single_model(
        model,
        criterion,
        optimizer,
        loaders,
        args,
        first_spike_fn=first_spike_fns,
        use_wandb=use_wandb,
        scheduler=scheduler,
    )
    if use_wandb:
        run.finish()
    return train_results


if __name__ == "__main__":
    use_wandb = False

    data_config = {
        "seed": np.random.randint(1000),
        "dataset": "ying_yang",
        "deterministic": False,
        "batch_size": 5,
        "encoding": "latency",
        "T": 30,
        "dt": 1e-3,
        "t_min": 2,
        "data_folder": "data",
        "input_dropout": None,
    }

    paper_params = {
        "mnist": {
            "mu": [0.078, 0.2],
            "sigma": [0.045, 0.37],
        },
        "ying_yang": {
            "mu": [1.5, 0.78],
            "sigma": [0.93, 0.1],
        },
    }

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
            "scale": 3.0,
            "mu": paper_params[data_config["dataset"]]["mu"],
            "sigma": paper_params[data_config["dataset"]]["sigma"],
            "n_hid": 100,
            "resolve_silent": False,
        },
        "device": torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu"),
    }

    training_config = {
        "n_epochs": 10,
        "loss": "ce_temporal",
        "alpha": 0.0,
        "xi": 1,
        "beta": 6.4,
    }

    optim_config = {"lr": 1e-2, "weight_decay": 0.0, "optimizer": "adam", "gamma": 0.9}

    config = {
        "data": data_config,
        "model": model_config,
        "training": training_config,
        "optim": optim_config,
    }
    flat_config = get_flat_dict_from_nested(config)
    train_results = main(flat_config, use_wandb=use_wandb)
