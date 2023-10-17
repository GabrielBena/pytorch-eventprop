import torch
import torch.nn.functional as F
from tqdm import tqdm
from snntorch.spikegen import rate, latency
import numpy as np


def encode_data(data, args):
    if not isinstance(data, torch.Tensor):
        data = torch.from_numpy(data)
    if "latency" in args.encoding:
        if "snntorch" in args.encoding:
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
            spike_data = spike_data.flatten(start_dim=2)
    return spike_data


def train(model, criterion, optimizer, loader, args, first_spike_fn=None):
    total_correct = 0.0
    total_loss = 0.0
    total_samples = 0.0
    model.train()

    pbar = tqdm(loader, leave=None, position=0)
    for batch_idx, (input, target) in enumerate(pbar):
        input, target = input.to(args.device), target.to(args.device)
        input = encode_data(input, args)

        total_correct = 0.0
        total_loss = 0.0
        total_samples = 0.0

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
        total_correct += predictions.eq(target.data.view_as(predictions)).sum().item()
        total_loss += loss.item() * len(target)
        total_samples += len(target)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        # if batch_idx % args.print_freq == 0:
        desc = "Batch {:03d}/{:03d}: Acc {:.2f}  Loss {:.3f} FR {}".format(
            batch_idx,
            len(loader),
            100 * total_correct / total_samples,
            total_loss / total_samples,
            np.round(np.array([s.data.cpu().numpy().mean() for s in all_spikes]), 2),
        )
        pbar.set_description(desc)
    # desc = ('\t\tTrain: \tAcc {:.2f}  Loss {:.3f}'.format(100*total_correct/total_samples, total_loss/total_samples))
    pbar.set_description(desc)


def test(model, loader, args, first_spike_fn=None):
    total_correct = 0.0
    total_samples = 0.0
    model.eval()

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(args.device), target.to(args.device)
            spike_data = encode_data(data, args)

            output, spikes = model(spike_data)
            if first_spike_fn:
                first_post_spikes = first_spike_fn(output)

            predictions = first_post_spikes.data.min(1, keepdim=True)[1]
            total_correct += (
                predictions.eq(target.data.view_as(predictions)).sum().item()
            )
            total_samples += len(target)

        print("\t\tTest: \tAcc {:.2f}".format(100 * total_correct / total_samples))
