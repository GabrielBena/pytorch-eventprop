from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
import numpy as np
import argparse
import wandb
import random

import torch
import torch.nn.functional as F

from snntorch.spikegen import rate, latency
from eventprop.config import get_flat_dict_from_nested


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


def get_label_outputs(output_matrix, labels):
    # Get the indices for all dimensions except the last one
    indices = np.indices(labels.shape)

    # Use advanced indexing to get the outputs of the label classes
    label_outputs = output_matrix[tuple(indices) + (labels,)]

    return label_outputs


def compute_accuracy(first_spike_times, labels):
    correct = first_spike_times.argmin(-1) == labels
    single_spiking = (
        np.count_nonzero(
            get_label_outputs(first_spike_times, labels)[..., None]
            == first_spike_times,
            axis=-1,
        )
        == 1
    )
    correct = np.logical_and(correct, single_spiking)
    return correct.mean(), correct


def encode_data(data, args):
    if not isinstance(data, torch.Tensor):
        data = torch.from_numpy(data)
    if "latency" in args.encoding:
        if not "eventprop" in args.encoding:
            spike_data = latency(
                data,
                args.T,
                first_spike_time=args.t_min,
                normalize=True,
                linear=True,
                interpolate=False,
            ).flatten(start_dim=2)
        else:
            if args.t_max is None:
                args.t_max = int(args.T * 3 / 5)
            spike_data = args.t_min + (args.t_max - args.t_min) * (data < 0.5).view(
                data.shape[0], -1
            )
            spike_data = F.one_hot(spike_data.long(), int(args.T)).permute(2, 0, 1)

        if args.dataset == "ying_yang":
            t0_spike = torch.zeros_like(spike_data[..., 0])
            t0_spike[0] = 1.0
            spike_data = torch.cat([spike_data, t0_spike.unsqueeze(-1)], dim=-1)

    else:
        spike_data = rate(data, args.T, gain=0.7)
        if len(spike_data.shape) > 2:
            spike_data = spike_data.flatten(start_dim=2).float()
    return spike_data


def train(
    model,
    criterion,
    optimizer,
    loader,
    args,
    first_spike_fn=None,
    pbar=None,
    scheduler=None,
):
    total_correct = 0.0
    total_loss = 0.0
    total_samples = 0.0
    model.train()
    n_last = max(256 // loader.batch_size, 1)

    if pbar is None:
        pbar_f = tqdm(loader, leave=None, position=0)
    else:
        pbar_f = loader

    total_correct = []
    total_loss = []
    total_samples = []

    for batch_idx, (input, target) in enumerate(pbar_f):
        input, target = input.to(args.device), target.to(args.device)
        input = encode_data(input, args)

        output, out_dict = model(input)
        if first_spike_fn:
            if isinstance(first_spike_fn, (list, tuple)):
                all_first_spikes = tuple(fn(output) for fn in first_spike_fn)
                first_spikes = all_first_spikes[0]
            else:
                first_spikes = first_spike_fn(output)
        else:
            first_spikes = output

        if isinstance(criterion, (list, tuple)):
            outputs = [output.clone() for _ in criterion]
            [o.retain_grad() for o in outputs]
            losses, spk_times = zip(*[c(o, target) for o, c in zip(outputs, criterion)])
            [l.retain_grad() for l in losses]
            [s.retain_grad() for s in spk_times]
            loss = sum(losses)
        else:
            loss = criterion(output, target)[0]

        if args.alpha != 0:
            target_first_spike_times = first_spikes.gather(1, target.view(-1, 1))
            loss += (
                args.alpha
                * (torch.exp(target_first_spike_times / (args.beta)) - 1).mean()
            )

        # predictions = first_spikes.data.min(-1, keepdim=True)[1]
        # total_correct.append(
        #     predictions.eq(target.data.view_as(predictions)).sum().item()
        # )
        accuracy, correct = compute_accuracy(
            first_spikes.cpu().detach().numpy(), target.cpu().detach().numpy()
        )
        total_correct.append(correct.sum())
        total_loss.append(loss.item() * len(target))
        total_samples.append(len(target))

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # if batch_idx % args.print_freq == 0:
        frs = np.round(
            np.array([s.data.cpu().numpy().mean() for s in out_dict["spikes"]]), 2
        )
        desc = "Batch {:03d}/{:03d}: Acc {:.2f}  Loss {:.3f} FR {}".format(
            batch_idx,
            len(loader),
            100
            * np.array(total_correct)[-n_last:].sum()
            / np.array(total_samples)[-n_last:].sum(),
            np.array(total_loss)[-n_last:].sum()
            / np.array(total_samples)[-n_last:].sum(),
            frs,
        )
        descs = pbar.desc.split("|")
        descs[0] = desc
        pbar.set_description_str(descs[0] + "|" + descs[1])
        mean_loss = np.array(total_loss).mean()
        mean_acc = np.array(total_correct).sum() / np.array(total_samples).sum()

    return mean_loss, mean_acc


def test(model, criterion, loader, args, first_spike_fn=None, pbar=None):
    total_correct = 0.0
    total_samples = 0.0
    total_loss = 0.0
    model.eval()

    total_spikes = 0.0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(args.device), target.to(args.device)
            spike_data = encode_data(data, args)

            output, spikes = model(spike_data)
            if first_spike_fn:
                if isinstance(first_spike_fn, (list, tuple)):
                    all_first_spikes = tuple(fn(output) for fn in first_spike_fn)
                    first_spikes = all_first_spikes[0]
                else:
                    first_spikes = first_spike_fn(output)
            else:
                first_spikes = output

            if isinstance(criterion, (list, tuple)):
                outputs = [output.clone() for _ in criterion]
                losses = [c(o, target)[0] for o, c in zip(outputs, criterion)]
                loss = losses[0]
            else:
                loss = criterion(output, target)[0]

            total_loss += loss
            # predictions = first_spikes.data.min(1, keepdim=True)[1]
            # total_correct += (
            #     predictions.eq(target.data.view_as(predictions)).sum().item()
            # )
            accuracy, correct = compute_accuracy(
                first_spikes.cpu().detach().numpy(), target.cpu().detach().numpy()
            )
            total_correct += correct.sum()
            total_samples += target.numel()
            total_spikes += output.sum().item()

        desc = "Test: Acc {:.2f}".format(100 * total_correct / total_samples)
        if pbar is None:
            print(desc)
        else:
            descs = pbar.desc.split("|")
            descs[1] = desc
            pbar.set_description_str(descs[0] + "|" + descs[1])

        test_acc = total_correct / total_samples
        test_loss = total_loss / len(loader)
        return test_loss, test_acc, total_spikes


def train_single_model(
    model,
    criterion,
    optimizer,
    loaders,
    args,
    first_spike_fn=None,
    use_tqdm=True,
    use_wandb=False,
    scheduler=None,
):
    n_epochs = getattr(args, "n_epochs", 2)
    pbar = range(n_epochs)
    train_accs, test_accs = [], []
    train_losses, test_losses = [], []
    best_loss = 1e10
    best_model = None

    if use_wandb:
        if wandb.run is None:
            need_finish = True
            run = wandb.init(project="eventprop", entity="m2snn", config=args)
        else:
            need_finish = False
            run = wandb.run
        # Default values overwritten by potential sweep
        args = argparse.Namespace(**get_flat_dict_from_nested(run.config))
    elif isinstance(args, dict):
        args = argparse.Namespace(**get_flat_dict_from_nested(args))

    if use_tqdm:
        tqdm_f = tqdm_notebook if is_notebook() else tqdm
        pbar = tqdm_f(pbar, leave=None, position=0, desc="|")
    for epoch in pbar:
        if epoch > 0:
            train_loss, train_acc = train(
                model,
                criterion,
                optimizer,
                loaders["train"],
                args,
                first_spike_fn=first_spike_fn,
                pbar=pbar,
                scheduler=scheduler,
            )
        else:
            train_loss, train_acc = None, None

        test_loss, test_acc, total_spikes = test(
            model,
            criterion,
            loaders["test"],
            args,
            first_spike_fn=first_spike_fn,
            pbar=pbar,
        )

        if (
            total_spikes == 0
            and epoch > 0
            and getattr(model, "model_type", "eventprop")
        ) == "eventprop":
            print("No spikes fired, stopping training")
            break

        train_accs.append(train_acc)
        train_losses.append(train_loss)
        test_accs.append(test_acc)
        test_losses.append(test_loss.cpu().data.item())

        if test_loss < best_loss:
            best_loss = test_loss
            best_model = model.state_dict()

        if use_wandb and epoch > 0:
            wandb.log(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                    "epoch": epoch,
                }
            )

    result_dict = {
        "train_acc": train_accs,
        "test_acc": test_accs,
        "train_loss": train_losses,
        "test_loss": test_losses,
        "best_model": best_model,
    }
    if use_wandb and need_finish:
        run.finish()

    return result_dict
