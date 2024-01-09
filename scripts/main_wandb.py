import numpy as np
import wandb
import random
import torch
import torch.nn.functional as F
import argparse
import os
from snntorch.functional.loss import (
    ce_temporal_loss,
    SpikeTime,
    ce_rate_loss,
    ce_count_loss,
)

from torchvision import datasets, transforms
from yingyang.dataset import YinYangDataset
from eventprop.models import SNN, SpikeCELoss
from eventprop.training import train_single_model
from eventprop.config import get_flat_dict_from_nested


def main(args, use_wandb=False, **override_params):
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

        config = wandb.config
        # Default values overwritten by potential sweep

    config.update(override_params, allow_val_change=True)

    # ------ Data ------

    config["dataset"] = config["dataset"]
    if config["deterministic"]:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"])
        random.seed(config["seed"])
        if use_wandb:
            wandb.log({"seed": config["seed"]})

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

        if config.get("subset_sizes", None) is not None:
            if config["subset_sizes"][0]:
                indices = np.concatenate(
                    [
                        np.random.choice(
                            np.where(train_dataset.targets == d)[0],
                            config["subset_sizes"][0] // 10,
                        )
                        for d in range(10)
                    ]
                )
                train_dataset = torch.utils.data.Subset(train_dataset, indices=indices)

            if config["subset_sizes"][1]:
                indices = np.concatenate(
                    [
                        np.random.choice(
                            np.where(test_dataset.targets == d)[0],
                            config["subset_sizes"][1] // 10,
                        )
                        for d in range(10)
                    ]
                )
                test_dataset = torch.utils.data.Subset(test_dataset, indices=indices)

    elif config["dataset"] == "ying_yang":
        train_dataset = YinYangDataset(size=100, seed=config["seed"])
        test_dataset = YinYangDataset(size=1000, seed=config["seed"] + 1)

    else:
        raise ValueError("Invalid dataset name")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=256, shuffle=False, drop_last=True
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
            optimizer, step_size=1, gamma=(config["gamma"]) ** (1 / len(train_loader))
        )
    else:
        scheduler = None
    loaders = {"train": train_loader, "test": test_loader}

    if config["loss"] == "ce_temporal":
        if config["model_type"] == "snntorch":
            criterion = ce_temporal_loss()
        elif config["model_type"] == "eventprop":
            criterion = SpikeCELoss(config["xi"])
        else:
            raise ValueError("Invalid model type")
    elif config["loss"] == "ce_both":
        criterion = [ce_temporal_loss(), SpikeCELoss(config["xi"])]
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

    return train_results


if __name__ == "__main__":
    use_wandb = False
    file_dir = os.path.dirname(os.path.abspath(__file__))

    data_config = {
        "seed": np.random.randint(10000),
        "dataset": "mnist",
        "subset_sizes": [300, 2560],
        "deterministic": True,
        "batch_size": 10,
        "encoding": "latency",
        "T": 30,
        "dt": 1e-3,
        "t_min": 2,
        "data_folder": f"{file_dir}/../../data",
        "input_dropout": 0.05,
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
            "scale": 3,
            "mu": paper_params[data_config["dataset"]]["mu"],
            "sigma": paper_params[data_config["dataset"]]["sigma"],
            "n_hid": 100,
            "resolve_silent": False,
            "dropout": 0.02,
        },
        "device": torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu"),
        # "device": "cpu",
    }

    training_config = {
        "n_epochs": 15,
        "loss": "ce_temporal",
        "alpha": 0.0,
        "xi": 1,
        "beta": 6.4,
        "n_tests": 1,
    }

    optim_config = {
        "lr": 1e-2,
        "weight_decay": 1e-6,
        "optimizer": "adam",
        "gamma": 0.9,
    }

    config = {
        "data": data_config,
        "model": model_config,
        "training": training_config,
        "optim": optim_config,
    }
    flat_config = get_flat_dict_from_nested(config)
    all_train_results = []
    all_seeds = []

    for test in range(training_config["n_tests"]):
        train_results = main(
            flat_config, use_wandb=use_wandb, seed=flat_config["seed"] + test
        )
        all_train_results.append(train_results)
        all_seeds.append(flat_config["seed"] + test)

    all_test_accs = np.array(
        [np.max(train_results["test_acc"][-3:]) for train_results in all_train_results]
    )
    all_test_losses = np.array(
        [np.min(train_results["test_loss"][-3:]) for train_results in all_train_results]
    )

    data = [
        [test_accs, test_losses, seed]
        for test_accs, test_losses, seed in zip(
            all_test_accs, all_test_losses, all_seeds
        )
    ]
    table = wandb.Table(data=data, columns=["test_acc", "test_loss", "seed"])

    if use_wandb:
        wandb.log(
            {
                "results_table": table,
                "mean_test_acc": np.mean(all_test_accs),
                "std_test_acc": np.std(all_test_accs),
                "mean_test_loss": np.mean(all_test_losses),
                "std_test_loss": np.std(all_test_losses),
            }
        )

        wandb.run.finish()

    print(
        f"Finished run, Mean test acc: {np.mean(all_test_accs)}, Mean test loss: {np.mean(all_test_losses)}"
    )
