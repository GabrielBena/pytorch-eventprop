# Meta-Training SNNs using MAML
import matplotlib.pyplot as plt
import numpy as np

import torch
from torchmeta.transforms import ClassSplitter
from eventprop.training import encode_data
from yingyang.meta_dataset import YingYangMetaDataset
from torchmeta.utils.data import BatchMetaDataLoader
from eventprop.config import get_flat_dict_from_nested

import random
from tqdm.notebook import tqdm
import pandas as pd
import seaborn as sns
import argparse

from eventprop.models import Meta_SNN, SpikeCELoss
from snn_maml.maml import ModelAgnosticMetaLearning

import wandb


if __name__ == "__main__":

    ## Data and Config

    data_config = {
        "seed": np.random.randint(0, 1000),
        "dataset": "ying_yang",
        "deterministic": True,
        "meta_batch_size": 10,
        "encoding": "latency",
        "T": 50,
        "dt": 1e-3,
        "t_min": 0,
        "t_max": -2,
        "data_folder": "../../../data/",
        "dataset_size": 1000,
        "n_tasks_per_split": 64,
    }

    data_args = argparse.Namespace(**data_config)

    torch.manual_seed(data_config["seed"])
    np.random.seed(data_config["seed"])
    random.seed(data_config["seed"])

    data_config["dataset"] = data_config["dataset"]
    if data_config["deterministic"]:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    encode_tranform = lambda s: (encode_data(s[0], data_args), s[1])
    dataset_split = lambda d: ClassSplitter(
        d,
        num_train_per_class=data_config["dataset_size"],
        num_test_per_class=data_config["dataset_size"],
        shuffle=False,
    )

    meta_train_dataset = YingYangMetaDataset(
        num_classes_per_task=1,
        meta_train=True,
        transform=encode_tranform,
        data_config=data_config,
        dataset_transform=dataset_split,
    )
    meta_val_dataset = YingYangMetaDataset(
        num_classes_per_task=1,
        meta_val=True,
        transform=encode_tranform,
        data_config=data_config,
        dataset_transform=dataset_split,
    )
    meta_test_dataset = YingYangMetaDataset(
        num_classes_per_task=1,
        meta_test=True,
        transform=encode_tranform,
        data_config=data_config,
        dataset_transform=dataset_split,
    )

    meta_train_dataloader = BatchMetaDataLoader(
        meta_train_dataset,
        data_config["meta_batch_size"],
        shuffle=True,
        num_workers=4,
    )

    meta_val_dataloader = BatchMetaDataLoader(
        meta_val_dataset,
        data_config["meta_batch_size"],
        shuffle=False,
        num_workers=4,
    )

    meta_test_dataloader = BatchMetaDataLoader(
        meta_test_dataset,
        data_config["meta_batch_size"],
        shuffle=False,
        num_workers=4,
    )

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
            # "scale_0_mu": 3,
            "scale": 6,
            "n_hid": 120,
            "resolve_silent": False,
            "dropout": 0.0,
        },
        # "device": (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")),
        "device": torch.device("cpu"),
    }

    n_ins = {"mnist": 784, "ying_yang": 5 if data_config["encoding"] == "latency" else 4}
    n_outs = {"mnist": 10, "ying_yang": 3}

    training_config = {
        "num_epochs": 100,
        "n_tests": 3,
        "exclude_equal": False,
        "do_train": True,
        "do_test": False,
        "test_every": 3,
    }

    loss_config = {
        "loss": "ce_temporal",
        "alpha": 3e-3,
        "xi": 0.5,
        "beta": 6.4,
    }

    maml_config = {
        "num_shots": 100,
        "num_shots_test": 1000,
        "first_order": False,
        "meta-lr": 1e-3,  # adam
        "inner-lr": 100,  # sgd
        "learn_step_size": False,
        "meta-gamma": 0.95,
    }

    default_config = {
        "data": data_config,
        "model": model_config,
        "training": training_config,
        "maml": maml_config,
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

    model = Meta_SNN(dims, **config).to(config["device"])

    meta_optimizer = torch.optim.Adam(model.parameters(), lr=maml_config["meta-lr"])
    if maml_config["meta-gamma"] is not None:
        meta_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            meta_optimizer, gamma=maml_config["meta-gamma"]
        )
    metalearner = ModelAgnosticMetaLearning(
        model,
        meta_optimizer,
        step_size=maml_config["inner-lr"],
        first_order=maml_config["first_order"],
        loss_function=SpikeCELoss(xi=config["xi"]),
        device=config["device"],
        num_adaptation_samples=maml_config["num_shots"],
    )

    all_test = np.zeros(args.num_epochs)
    all_train = np.zeros(args.num_epochs)
    all_vals = np.zeros(args.num_epochs)
    epoch_desc = "Epoch {{0: <{0}d}}".format(1 + int(np.log10(args.num_epochs)))
    results_accuracy_after = []

    # -------------------------------
    # -------- Main Training --------
    # -------------------------------

    use_wandb = False

    if use_wandb:
        if wandb.run is None:
            run = wandb.init(project="snn-maml", config=config, entity="m2snn")
        else:
            run = wandb.run
        config = wandb.config
        # Default values overwritten by potential sweep

    for epoch in range(args.num_epochs):
        # print(epoch, meta_scheduler.get_last_lr())
        if args.do_train:
            results_train = metalearner.train(
                meta_train_dataloader,
                max_batches=1,
                desc="Training",
                leave=False,
                epoch=epoch,
            )  # ,
            # deltaw=args.deltaw)

            if results_train is not None:
                all_train[epoch] = np.mean(results_train["accuracies_after"])
                if use_wandb:
                    wandb.log(
                        {
                            "train_accuracies_after": results_train["accuracies_after"],
                            "train_accuracies_before": results_train["accuracies_before"],
                            "train_outer_loss": results_train["mean_outer_loss"],
                            "epoch": epoch,
                        }
                    )

        if epoch % args.test_every == 0:
            results = metalearner.evaluate(
                meta_val_dataloader,
                max_batches=5,
                desc=epoch_desc.format(epoch + 1),
            )

            if "accuracies_after" in results:
                results_accuracy_after.append(results["accuracies_after"])
                if use_wandb:
                    wandb.log(
                        {
                            "val_accuracies_after": results["accuracies_after"],
                            "val_accuracies_before": results["accuracies_before"],
                            "val_outer_loss": results["mean_outer_loss"],
                            "epoch": epoch,
                        }
                    )

        if args.do_test and args.test_every > 0 and epoch % args.test_every == 0:
            results_test = metalearner.evaluate(
                meta_test_dataloader,
                max_batches=10,
                desc=epoch_desc.format(epoch + 1),
            )  # ,
            # deltaw=args.deltaw)

            if use_wandb:
                wandb.log(
                    {
                        "test_accuracies_after": results["accuracies_after"],
                        "test_accuracies_before": results["accuracies_before"],
                        "test_outer_loss": results["mean_outer_loss"],
                    }
                )

            all_test[epoch] = np.mean(results_test["accuracies_after"])
