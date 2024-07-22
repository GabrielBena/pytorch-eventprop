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

from torch.utils.data.dataset import ConcatDataset
from torchvision import datasets, transforms
from yingyang.dataset import YinYangDataset
from eventprop.models import SNN, SpikeCELoss, FirstSpikeTime, SpikeQuadLoss
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

    print("Overriding config with:", override_params)
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

        train_dataset = YinYangDataset(
            size=config["subset_sizes"][0] if config["subset_sizes"][0] else 5000,
            seed=config["seed"],
        )
        test_dataset = ConcatDataset(
            [
                YinYangDataset(
                    size=(
                        config["subset_sizes"][1] if config["subset_sizes"][1] else 2000
                    ),
                    seed=config["seed"] + i,
                )
                for i in range(1, 5)
            ],
        )

    else:
        raise ValueError("Invalid dataset name")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=(
            min(256, config["subset_sizes"][1]) if config["subset_sizes"][1] else 256
        ),
        shuffle=False,
        drop_last=True,
    )

    # ------ Model ------

    n_ins = {"mnist": 784, "ying_yang": 5 if config["encoding"] == "latency" else 4}
    n_outs = {"mnist": 10, "ying_yang": 3}

    # config.update(
    #     {"tau_s": min(config["tau_s"], config["tau_m"])}, allow_val_change=True
    # )

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
    try:
        optimizer = optimizers_type[config["optimizer"]](
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            betas=(config["adam_beta_1"], config["adam_beta_2"]),
        )
    except ValueError:
        optimizer = optimizers_type[config["optimizer"]](
            model.parameters(), config["lr"]
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
            criterion = SpikeCELoss(config["xi"], config["alpha"], config["beta"])
        else:
            raise ValueError("Invalid model type")
    elif config["loss"] == "quadratic":
        criterion = SpikeQuadLoss(config["xi"], config["alpha"], config["beta"])

    elif config["loss"] == "ce_rate":
        criterion = ce_rate_loss()
    elif config["loss"] == "ce_count":
        criterion = ce_count_loss()
    else:
        raise ValueError("Invalid loss type")

    first_spike_fn = FirstSpikeTime.apply

    args = argparse.Namespace(**config)
    train_results = train_single_model(
        model,
        criterion,
        optimizer,
        loaders,
        args,
        first_spike_fn=first_spike_fn,
        use_wandb=use_wandb,
        scheduler=scheduler,
    )

    return train_results


if __name__ == "__main__":

    use_wandb = False
    file_dir = os.path.dirname(os.path.abspath(__file__))

    sweep_id = "s5hcchgl"
    use_best_params = False
    best_params_to_use = {"optim", "model"}
    # best_params_to_use = None
    use_run_params = False

    data_config = {
        "seed": np.random.randint(10000),
        # "seed": 4424,
        "dataset": "ying_yang",
        "subset_sizes": [5000, 1000],
        "deterministic": True,
        "batch_size": 22,
        "encoding": "latency",
        "T": 27,
        "dt": 1e-3,
        "t_min": 2,
        "t_max": None,
        "data_folder": f"{file_dir}/../../data",
        "input_dropout": 0.0,
    }

    paper_params = {
        "mnist": {
            "mu": [0.078, 0.2],
            "sigma": [0.045, 0.37],
        },
        "ying_yang": {
            "mu": [1.5, 0.93],
            "sigma": [0.78, 0.1],
        },
        "ying_yang_BS2": {"mu": [1.0, 0.4], "sigma": [0.01, 0.1]},
        "ying_yang_timo": {
            "mu": [1.5 * 2, 0.93 * 2],
            "sigma": [0.78 * 2, 0.1 * 2],
        },
    }

    model_config = {
        "model_type": "eventprop",
        "snn": {
            "dt": data_config["dt"],
            "tau_m": 20e-3,
            "tau_s": 5e-3,
        },
        "weights": {
            # "init_mode": "kaiming_both",
            "init_mode": "paper",
            # Used in case of "kaiming_both" init_mode
            "scales": {
                0: {
                    "scale_0_mu": 3.2,
                    "scale_0_sigma": 3.2,
                },
                1: {
                    "scale_1_mu": 5.2,
                    "scale_1_sigma": 2.8,
                },
            },
            # Used in case of "paper" init_mode
            "n_hid": 120,
            "resolve_silent": False,
            "dropout": 0.0,
        },
        "device": torch.device("cpu"),
    }

    model_config["weights"]["distribution"] = (
        paper_params["ying_yang_timo"]
        if "paper" in model_config["weights"]["init_mode"] == "paper"
        else None
    )

    training_config = {
        "n_epochs": 60,
        "n_tests": 15,
        "exclude_equal": False,
    }

    optim_config = {
        "optimizer": "adam",
        # "lr": 0.0019,
        "lr": 5e-2,
        # "weight_decay": 0.00000065,
        "weight_decay": 0.0,
        "gamma": 0.95,  # decay per epoch
        "adam_beta_1": 0.9,
        "adam_beta_2": 0.999,
    }

    loss_config = {
        # "loss": "quadratic",
        "loss": "ce_temporal",
        "alpha": 0.0096,
        "xi": 1.43,
        "beta": 96,
    }

    config = {
        "data": data_config,
        "model": model_config,
        "training": training_config,
        "optim": optim_config,
        "loss": loss_config,
    }

    flat_config = get_flat_dict_from_nested(config)
    all_train_results = []
    all_seeds = []

    if use_best_params:
        api = wandb.Api()
        sweep_path = f"m2snn/eventprop/{sweep_id}"
        sweep = api.sweep(sweep_path)
        best_run = sweep.best_run()
        best_params = best_run.config
    elif use_run_params:
        api = wandb.Api()
        run = api.run(f"m2snn/eventprop/{use_run_params}")
        best_params = run.config
    else:
        best_params = {}

    if use_best_params and best_params_to_use is not None:
        best_params = {
            k: best_params[k]
            for k in get_flat_dict_from_nested(
                {k: config[k] for k in best_params_to_use}
            )
        }

    if "seed" in best_params:
        best_params.pop("seed")
    if "device" in best_params:
        best_params.pop("device")

    for test in range(training_config["n_tests"]):
        train_results = main(
            flat_config,
            use_wandb=use_wandb,
            seed=flat_config["seed"] + test,
            **best_params,
        )
        all_train_results.append(train_results)
        all_seeds.append(flat_config["seed"] + test)

    try:

        all_test_accs = np.array(
            [train_results["test_acc"] for train_results in all_train_results]
        )
        all_test_losses = np.array(
            [train_results["test_loss"] for train_results in all_train_results]
        )

    except ValueError:
        all_test_accs = np.array(
            [train_results["test_acc"] for train_results in all_train_results],
            dtype=object,
        )
        all_test_losses = np.array(
            [train_results["test_loss"] for train_results in all_train_results],
            dtype=object,
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
                "mean_test_acc": np.mean(all_test_accs[:, -1]),
                "max_test_acc": np.mean(np.max(all_test_accs, axis=1)),
                "max_last3_test_acc": np.mean(np.max(all_test_accs[:, -3:], axis=1)),
                "std_last_test_acc": np.std(all_test_accs[:, -1]),
                "last_test_loss": np.mean(all_test_losses[:, -1]),
                "min_test_loss": np.mean(np.min(all_test_losses, axis=1)),
                "min_last3_test_loss": np.mean(np.min(all_test_losses[:, -3:], axis=1)),
                "std_last_test_loss": np.std(all_test_losses[:, -1]),
            }
        )

        wandb.run.finish()

    print(
        str(
            f"Finished run, Mean test acc: {np.mean(all_test_accs[:, -1])}"
            + f"Mean test loss: {np.mean(all_test_losses)}"
        )
    )
