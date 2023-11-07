import torch
import torch.nn.functional as F
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
from snntorch.spikegen import rate, latency
import numpy as np


def is_notebook():
    """
    Check if the code is running in a Jupyter notebook or not.

    Returns:
    -------
    bool
        True if running in a notebook, False otherwise.
    """
    try:
        get_ipython()
        notebook = True
    except NameError:
        notebook = False
    return notebook


def encode_data(data, config):
    if not isinstance(data, torch.Tensor):
        data = torch.from_numpy(data)
    if "latency" in config["encoding"]:
        if not "eventprop" in config["encoding"]:
            spike_data = latency(
                data,
                config["T"],
                first_spike_time=config["t_min"],
                normalize=True,
                linear=True,
                interpolate=False,
            ).flatten(start_dim=2)
        else:
            if config["t_max"] is None:
                config["t_max"] = int(config["T"] * 3 / 5)
            spike_data = config["t_min"] + (config["t_max"] - config["t_min"]) * (
                data < 0.5
            ).view(data.shape[0], -1)
            spike_data = F.one_hot(spike_data.long(), int(config["T"])).permute(2, 0, 1)

        if config["dataset"] == "ying_yang" and False:
            t0_spike = torch.zeros_like(spike_data[..., 0])
            t0_spike[0] = 1.0
            spike_data = torch.cat([spike_data, t0_spike.unsqueeze(-1)], dim=-1)

    else:
        spike_data = rate(data, config["T"], gain=0.7)
        if len(spike_data.shape) > 2:
            spike_data = spike_data.flatten(start_dim=2).float()
    return spike_data


def train(model, criterion, optimizer, loader, args, first_spike_fn=None, pbar=None):
    total_correct = 0.0
    total_loss = 0.0
    total_samples = 0.0
    model.train()

    if pbar is None:
        pbar_f = tqdm(loader, leave=None, position=0)
    else:
        pbar_f = loader

    total_correct = []
    total_loss = []
    total_samples = []

    for batch_idx, (input, target) in enumerate(pbar_f):
        input, target = input.to(args.device), target.to(args.device)
        input = encode_data(input, args.config)

        output, all_spikes = model(input)
        if first_spike_fn:
            first_spikes = first_spike_fn(output)
        else:
            first_spikes = output

        loss = criterion(output, target)

        if args.alpha != 0:
            target_first_spike_times = first_spikes.gather(1, target.view(-1, 1))
            loss += (
                args.alpha
                * (
                    torch.exp(target_first_spike_times / (args.beta * args.tau_s)) - 1
                ).mean()
            )

        predictions = first_spikes.data.min(-1, keepdim=True)[1]
        total_correct.append(
            predictions.eq(target.data.view_as(predictions)).sum().item()
        )
        total_loss.append(loss.item() * len(target))
        total_samples.append(len(target))

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        # if batch_idx % args.print_freq == 0:
        desc = "Batch {:03d}/{:03d}: Acc {:.2f}  Loss {:.3f} FR {} |".format(
            batch_idx,
            len(loader),
            100
            * np.array(total_correct)[-10:].sum()
            / np.array(total_samples)[-10:].sum(),
            np.array(total_loss)[-10:].sum() / np.array(total_samples)[-10:].sum(),
            np.round(np.array([s[0].data.cpu().numpy().mean() for s in all_spikes]), 2),
        )
        descs = pbar.desc.split("|")
        descs[0] = desc
        pbar.set_description(descs[0] + " | " + descs[1])

    return total_correct, total_loss, total_samples


def test(model, loader, args, first_spike_fn=None, pbar=None):
    total_correct = 0.0
    total_samples = 0.0
    model.eval()

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(args.device), target.to(args.device)
            spike_data = encode_data(data, args.config)

            output, spikes = model(spike_data)
            if first_spike_fn:
                first_post_spikes = first_spike_fn(output)
            else:
                first_post_spikes = output

            predictions = first_post_spikes.data.min(1, keepdim=True)[1]
            total_correct += (
                predictions.eq(target.data.view_as(predictions)).sum().item()
            )
            total_samples += len(target)

        desc = "Test: \tAcc {:.2f}".format(100 * total_correct / total_samples)
        if pbar is None:
            print(desc)
        else:
            descs = pbar.desc.split("|")
            descs[1] = desc
            pbar.set_description(descs[0] + " | " + descs[1])


def train_model(
    model, criterion, optimizer, loaders, args, first_spike_fn=None, use_tqdm=True
):
    n_epochs = getattr(args, "n_epochs", 2)
    pbar = range(n_epochs)
    if use_tqdm:
        tqdm_f = tqdm_notebook if is_notebook() else tqdm
        pbar = tqdm_f(pbar, leave=None, position=0, desc=" | ")
    for epoch in pbar:
        if epoch > 1:
            train(
                model,
                criterion,
                optimizer,
                loaders["train"],
                args,
                first_spike_fn=first_spike_fn,
                pbar=pbar,
            )
        test(model, loaders["test"], args, first_spike_fn=first_spike_fn, pbar=pbar)
