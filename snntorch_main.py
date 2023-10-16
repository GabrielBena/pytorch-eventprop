import argparse, torch, random
import numpy as np
from torchvision import datasets, transforms
import torch.nn.functional as F
from models import SNN2, SpikeCELoss
from tqdm import tqdm
from yingyang.dataset import YinYangDataset

from snntorch.spikegen import rate
import torch.nn as nn
import snntorch as snn

parser = argparse.ArgumentParser(description="Training a SNN on MNIST with SNNTprch")

# General settings
parser.add_argument(
    "--data-folder",
    type=str,
    default="data",
    help="name of folder to place dataset (default: data)",
)
parser.add_argument(
    "--device", type=str, default="cpu", help="device to run on (default: cuda)"
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
    "--epochs", type=int, default=5, help="number of epochs to train (default: 100)"
)
parser.add_argument(
    "--lr", type=float, default=1., help="learning rate (default: 1.0)"
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=128,
    help="size of batch used for each update step (default: 128)",
)

parser.add_argument(
    "--dataset",
    type=str,
    default="ying_yang",
    help="dataset to use (default: mnist)",
)

# Loss settings (specific for SNNs)
parser.add_argument(
    "--xi",
    type=float,
    default=0.4,
    help="constant factor for cross-entropy loss (default: 0.4)",
)
parser.add_argument(
    "--alpha",
    type=float,
    default=0.01,
    help="regularization factor for early-spiking (default: 0.01)",
)
parser.add_argument(
    "--beta",
    type=float,
    default=2,
    help="constant factor for regularization term (default: 2.0)",
)

# Spiking Model settings
parser.add_argument(
    "--T",
    type=float,
    default=20,
    help="duration for each simulation, in ms (default: 20)",
)
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
    "--mu",
    type=float,
    default=np.array([1.0, 1]) * 0.1,
    help="factor to scale the weights (default: 1.0)",
)

parser.add_argument(
    "--t_max",
    type=float,
    default=12.0,
    help="max input spiking time, in ms (default: 12)",
)
parser.add_argument(
    "--t_min",
    type=float,
    default=2.0,
    help="min input spiking time, in ms (default: 2)",
)

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

if args.deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# def encode_data(data):
#     spike_data = args.t_min + (args.t_max - args.t_min) * (data < 0.5).view(
#         data.shape[0], -1
#     )
#     spike_data = F.one_hot(spike_data.long(), int(args.T)).permute(2, 0, 1).float()
#     return spike_data

def encode_data(data) : 
    spike_data = rate(data, args.T, gain=0.75).flatten(start_dim=2).float()
    return spike_data
    # return data.flatten(start_dim=2).transpose(0, 1)

# Network Architecture


# Temporal Dynamics
num_steps = args.T
beta = 0.95
# Define Network
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, input):

        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for step, x in enumerate(input):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0).squeeze(), torch.stack(mem2_rec, dim=0).squeeze()
        
# Load the network onto CUDA if available

def train(model, criterion, optimizer, loader):
    total_correct = 0.0
    total_loss = 0.0
    total_samples = 0.0
    model.train()

    pbar = tqdm(loader, leave=None, position=0)
    for batch_idx, (input, target) in enumerate(pbar):
        input, target = input.to(args.device), target.to(args.device)
        input = encode_data(input)

        total_correct = 0.0
        total_loss = 0.0
        total_samples = 0.0

        spikes, mem = model(input)
        # output = F.softmax(mem.sum(0), dim=-1)
        output = spikes.sum(0)

        # loss = nn.CrossEntropyLoss()(output, target)
        loss = criterion(output, target)

        if args.alpha != 0:
            target_first_spike_times = output.gather(1, target.view(-1, 1))
            loss += (
                args.alpha
                * (
                    torch.exp(target_first_spike_times / (args.beta * args.tau_s)) - 1
                ).mean()
            )

        predictions = output.data.min(1, keepdim=True)[1]
        total_correct += predictions.eq(target.data.view_as(predictions)).sum().item()
        total_loss += loss.item() * len(target)
        total_samples += len(target)

        optimizer.zero_grad()
        loss.backward(retain_graph=False)

        optimizer.step()

        # if batch_idx % args.print_freq == 0:
        desc = "Batch {:03d}/{:03d}: Acc {:.2f}  Loss {:.3f} FR".format(
            batch_idx,
            len(loader),
            100 * total_correct / total_samples,
            total_loss / total_samples,
            # np.round(np.array([s.data.cpu().numpy().mean() for s in other[-1][:-1]]), 2),
        )
        pbar.set_description(desc)
    # desc = ('\t\tTrain: \tAcc {:.2f}  Loss {:.3f}'.format(100*total_correct/total_samples, total_loss/total_samples))
    pbar.set_description(desc)


def test(model, loader):
    total_correct = 0.0
    total_samples = 0.0
    model.eval()

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(args.device), target.to(args.device)
            spike_data = encode_data(data)

            first_post_spikes, spikes = model(spike_data)
            predictions = first_post_spikes.data.min(1, keepdim=True)[1]
            total_correct += (
                predictions.eq(target.data.view_as(predictions)).sum().item()
            )
            total_samples += len(target)

        print("\t\tTest: \tAcc {:.2f}".format(100 * total_correct / total_samples))


if args.dataset == "mnist":
    train_dataset = datasets.MNIST(
        args.data_folder, train=True, download=True, transform=transforms.ToTensor()
    )
    test_dataset = datasets.MNIST(
        args.data_folder, train=False, download=True, transform=transforms.ToTensor()
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
    test_dataset, batch_size=args.batch_size, shuffle=False
)

model_kwars = {
    "T": args.T,
    "dt": args.dt,
    "tau_m": args.tau_m,
    "tau_s": args.tau_s,
    "mu": args.mu,
}

n_ins = {"mnist": 784, "ying_yang": 4}
n_outs = {"mnist": 10, "ying_yang": 3}
n_hid = 30

dims = [n_ins[args.dataset]]
if n_hid is not None and isinstance(n_hid, list):
    dims.extend(n_hid)
elif isinstance(n_hid, int):
    dims.append(n_hid)
dims.append(n_outs[args.dataset])

num_inputs, num_hidden, num_outputs = dims

# model = SNN2(dims, **model_kwars).to(args.device)
model = Net().to(args.device)

criterion = SpikeCELoss(args.T, args.xi, args.tau_s)
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# model(encode_data(next(iter(train_loader))[0].to(args.device)))

for epoch in range(args.epochs):
    print("Epoch {:03d}/{:03d}".format(epoch, args.epochs))
    train(model, criterion, optimizer, train_loader)
    test(model, test_loader)
    scheduler.step()
