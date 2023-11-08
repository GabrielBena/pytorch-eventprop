import argparse, torch, random
import numpy as np
from torchvision import datasets, transforms
import os

from snntorch.functional.loss import (
    ce_temporal_loss,
    SpikeTime,
    ce_rate_loss,
    ce_count_loss,
)
from yingyang.dataset import YinYangDataset

from eventprop.models import SNN
from eventprop.training import train, test, encode_data
from eventprop.initalization import (
    FluctuationDrivenCenteredNormalInitializer,
)
import yaml, pyaml

if __name__ == "__main__":
    # %% Args

    parser = argparse.ArgumentParser(
        description="Training a SNN on MNIST with EventProp"
    )

    # General settings
    parser.add_argument(
        "--data-folder",
        type=str,
        default="~/SpiNNCloud/Code/data/",
        help="name of folder to place dataset (default: data)",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    parser.add_argument(
        "--print-freq",
        type=int,
        default=100,
        help="training stats are printed every so many batches (default: 100)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="run in deterministic mode for reproducibility",
    )

    # Training settings
    parser.add_argument(
        "--epochs", type=int, default=2, help="number of epochs to train (default: 100)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-2, help="learning rate (default: 1.0)"
    )
    parser.add_argument(
        "--optimizer", type=str, default="adam", help="optimizer to use (default: adam)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="size of batch used for each update step (default: 128)",
    )

    # Loss settings (specific for SNNs)
    parser.add_argument(
        "--xi",
        type=float,
        default=0.1,
        help="constant factor for cross-entropy loss (default: 0.4)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=3e-3,
        help="regularization factor for early-spiking (default: 0.01)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=6.4 / 5,
        help="constant factor for regularization term (default: 2.0)",
    )

    # Spiking Model settings

    parser.add_argument(
        "--dt",
        type=float,
        default=1,
        help="time step to discretize the simulation, in ms (default: 1)",
    )
    parser.add_argument(
        "--tau_m",
        type=float,
        default=20.0,
        help="membrane time constant, in ms (default: 20)",
    )
    parser.add_argument(
        "--tau_s",
        type=float,
        default=5.0,
        help="synaptic time constant, in ms (default: 5)",
    )

    parser.add_argument(
        "--t_max",
        type=object,
        default=None,
        help="max input spiking time, in ms (default: 12)",
    )

    parser.add_argument(
        "--t_min",
        type=float,
        default=2.0,
        help="min input spiking time, in ms (default: 2)",
    )
    # %% Main Args

    # ------ Main Arguments to change ------#
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        help="dataset to use (default: mnist)",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="latency_snntorch",
        help="type of spike encoding for the input",
    )
    parser.add_argument(
        "--T",
        type=float,
        default=30,
        help="duration for each simulation, in ms (default: 20)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="eventprop",
        help="model to use (default: eventprop)",
    )
    parser.add_argument(
        "--n_hid",
        type=object,
        default=100,
        help="number of hidden neurons (default: None)",
    )
    parser.add_argument(
        "--mu",
        type=float,
        # default=np.array([1]) * 0.1,
        default=[1, 1.0],
        help="factor to scale the weights (default: 0.1)",
    )

    parser.add_argument(
        "--loss_type",
        type=str,
        default="ce_temporal",
        help="loss function to use (default: ce_temporal)",
    )

    parser.add_argument(
        "--resolve_silent",
        action="store_false",
        help="add weight to fix silent neurons (default: True)",
    )

    parser.add_argument(
        "--use_fluct_init",
        action="store_true",
        help="use fluctuation init (default: False)",
    )

    parser.add_argument(
        "--simple_lif",
        action="store_true",
        help="use simple LIF neuron (default: False)",
    )

    args = parser.parse_args()
    print(args)

    current_dir = os.path.dirname(os.path.realpath(__file__))
    with open(current_dir + "/args.yaml", "w") as f:
        pyaml.dump(args.__dict__, f)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # %% Data

    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if args.dataset == "mnist":
        train_dataset = datasets.MNIST(
            args.data_folder, train=True, download=True, transform=transforms.ToTensor()
        )
        test_dataset = datasets.MNIST(
            args.data_folder,
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )
    elif args.dataset == "ying_yang":
        train_dataset = YinYangDataset(size=60000, seed=42)
        test_dataset = YinYangDataset(size=10000, seed=40)

    else:
        raise ValueError("Invalid dataset name")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False
    )

    # %% Model

    model_kwars = {
        "T": args.T,
        "dt": args.dt,
        "tau_m": args.tau_m,
        "tau_s": args.tau_s,
        "mu": args.mu,
        "resolve_silent": args.resolve_silent,
    }

    n_ins = {"mnist": 784, "ying_yang": 5 if "latency" in args.encoding else 4}
    n_outs = {"mnist": 10, "ying_yang": 3}

    dims = [n_ins[args.dataset]]
    if args.n_hid is not None and isinstance(args.n_hid, list):
        dims.extend(args.n_hid)
    elif isinstance(args.n_hid, int):
        dims.append(args.n_hid)
    dims.append(n_outs[args.dataset])

    model = (
        SNN(dims, **model_kwars).to(device)
        if args.model == "eventprop"
        else SNN2(dims, **model_kwars).to(device)
    )
    # %% Loss and Optimizer

    # criterion = SpikeCELoss(args.T, args.xi, args.tau_s)
    if args.loss_type == "ce_temporal":
        if args.model == "snntorch":
            criterion = ce_temporal_loss()
        elif args.model == "eventprop":
            criterion = ce_temporal_loss()
            # criterion.spk_time_fn.first_spike_fn = FirstSpikeTime.apply
            # criterion = SpikeCELoss(args.xi, args.tau_s)
        else:
            raise ValueError("Invalid model name")
    elif args.loss_type == "ce_rate":
        criterion = ce_rate_loss()
    elif args.loss_type == "ce_count":
        criterion = ce_count_loss()
    else:
        raise ValueError("Invalid loss type")

    first_spike_fn = SpikeTime().first_spike_fn
    # first_spike_fn = FirstSpikeTime.apply

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer in ["sgd", "SGD"]:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    else:
        raise ValueError("Invalid optimizer name")

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    # %% Init

    if args.use_fluct_init:
        data, target = next(iter(train_loader))
        spikes = encode_data(data.to(args.device), args)
        nu = spikes.sum(0).mean() / (args.T * args.dt * 1e-3)
        initializer = FluctuationDrivenCenteredNormalInitializer(
            sigma_u=1,
            nu=nu,
            timestep=args.dt * 1e-3,
        )
        initializer.initialize(model)

    # %% Training

    for epoch in range(args.epochs):
        print("Epoch {:03d}/{:03d}".format(epoch, args.epochs))
        if epoch > 0:
            train(
                model,
                criterion,
                optimizer,
                train_loader,
                args,
                first_spike_fn=first_spike_fn,
            )
            scheduler.step()
        test(model, test_loader, args, first_spike_fn=first_spike_fn)


# %%
