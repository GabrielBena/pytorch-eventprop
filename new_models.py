import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
import snntorch as snn
from snntorch import utils
from snntorch.functional import SpikeTime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reset(net) : 
    for m in net.modules() : 
        if hasattr(m, 'reset_hidden') : 
            m.reset_hidden()

def init_weights(w, weight_scale=1.0):
    k = weight_scale * (1.0 / np.sqrt(w.shape[1]))
    nn.init.uniform_(w, -k, k)
    print(k)


class WrapperFunction(Function):
    @staticmethod
    def forward(ctx, input, params, forward, backward):
        ctx.backward = backward
        pack, output = forward(input)
        ctx.save_for_backward(*pack)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        backward = ctx.backward
        pack = ctx.saved_tensors
        grad_input, grad_weight = backward(grad_output, *pack)
        return grad_input, grad_weight, None, None


class FirstSpikeTime(Function):
    @staticmethod
    def forward(ctx, input, t=None):
        idx = (
            torch.arange(input.shape[0], 0, -1)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .float()
            .to(device)
        )
        first_spike_times = torch.argmax(idx * input, dim=0).float()
        ctx.save_for_backward(input, first_spike_times.clone())
        first_spike_times[first_spike_times == 0] = input.shape[0] - 1
        return first_spike_times

    @staticmethod
    def backward(ctx, grad_output):
        input, first_spike_times = ctx.saved_tensors
        k = F.one_hot(first_spike_times.long(), input.shape[0]).float().permute(2, 0, 1)
        grad_input = k * grad_output.unsqueeze(0)
        return grad_input, None


class RecordingSequential(nn.Sequential):
    def __init__(self, *modules):
        super(RecordingSequential, self).__init__(*modules)

    def forward(self, x):
        recs = []
        for module in self._modules.values():
            x = module(x)
            recs.append(x)
            if isinstance(x, tuple):
                x = x[0]
        return x, recs


class SpikingLinear(nn.Module):
    def __init__(self, d1, d2, T, dt, tau_m, tau_s, mu, resolve_silent=False):
        super(SpikingLinear, self).__init__()

        self.input_dim = d1
        self.output_dim = d2
        self.T = T
        self.dt = dt
        self.tau_m = tau_m
        self.tau_s = tau_s
        self.resolve_silent = resolve_silent
        self.mu = mu if mu is not None else 1
        self.mu_silent = 1/np.sqrt(d1)

        self.weight = nn.Parameter(torch.Tensor(d2, d1))
        nn.init.kaiming_normal_(self.weight)
        # nn.init.normal_(self.weight, 0.1, 0.1)
        self.weight.data *= self.mu

        self.forward = lambda input: WrapperFunction.apply(
            input, self.weight, self.manual_forward, self.manual_backward
        )

    def __repr__(self):
        return super().__repr__()[:-1] + f"{self.input_dim}, {self.output_dim})"

    def manual_forward(self, input):
        steps = int(self.T / self.dt)

        V = torch.zeros(steps, input.shape[1], self.output_dim).to(device)
        I = torch.zeros(steps, input.shape[1], self.output_dim).to(device)
        output = torch.zeros(steps, input.shape[1], self.output_dim).to(device)

        while True:
            for i in range(1, steps):
                t = i * self.dt

                V[i] = (1 - self.dt / self.tau_m) * V[i - 1] + (
                    self.dt / self.tau_m
                ) * I[i - 1]

                # V[i] = (1 - self.dt / self.tau_m) * V[i - 1] + I[i - 1]

                I[i] = (1 - self.dt / self.tau_s) * I[i - 1] + F.linear(
                    input[i - 1].float(), self.weight
                )

                spikes = (V[i] > 1.0).float()
                output[i] = spikes
                V[i] = (1 - spikes) * V[i]

            if self.training and self.resolve_silent:
                is_silent = output.sum(0).mean(0) == 0
                self.weight.data[is_silent] += self.mu_silent
                if is_silent.sum() == 0:
                    break
            else : 
                break                    

        return (input, I, output), output

    def manual_backward(self, grad_output, input, I, post_spikes):
        steps = int(self.T / self.dt)

        lV = torch.zeros(steps, input.shape[1], self.output_dim).to(device)
        lI = torch.zeros(steps, input.shape[1], self.output_dim).to(device)

        grad_input = torch.zeros(steps, input.shape[1], input.shape[2]).to(device)
        grad_weight = torch.zeros(input.shape[1], *self.weight.shape).to(device)

        for i in range(steps - 2, -1, -1):
            t = i * self.dt

            delta = lV[i + 1] - lI[i + 1]
            grad_input[i] = F.linear(delta, self.weight.t())

            lV[i] = (1 - self.dt / self.tau_m) * lV[i + 1] + post_spikes[i + 1] * (
                lV[i + 1] + grad_output[i + 1]
            ) / (I[i] - 1 + 1e-10)

            lI[i] = lI[i + 1] + (self.dt / self.tau_s) * delta

            spike_bool = input[i].float()
            grad_weight -= spike_bool.unsqueeze(1) * lI[i].unsqueeze(2)

        return grad_input, grad_weight


class SpikingLinear2(nn.Module):
    def __init__(self, d1, d2, **kwargs) -> None:
        super().__init__()
        self.input_dim, self.output_dim = d1, d2
        self.mu = kwargs.get("mu", 1.)
        self.mu_silent = 1/np.sqrt(d1)
        self.resolve_silent = kwargs.get('resolve_silent', False)

        self.weight = nn.Parameter(torch.Tensor(d2, d1))
        nn.init.kaiming_normal_(self.weight)
        self.weight.data *= self.mu


        self.syn = snn.Synaptic(
            alpha=1 - kwargs["dt"] / kwargs["tau_s"],
            beta=1 - kwargs["dt"] / kwargs["tau_m"],
            init_hidden=True,
            reset_mechanism="zero",
            # output=d2 == dims[-1],
        )

        self.fix_silent = True

    def forward(self, input):
        while True : 
            out_spikes = []
            for t, in_spikes in enumerate(input):
                    
                out = F.linear(in_spikes, self.weight)
                spikes = self.syn(out)
                out_spikes.append(spikes)

            out_spikes = torch.stack(out_spikes, dim=0)
            if self.training and self.resolve_silent:
                is_silent = out_spikes.sum(0).mean(0) == 0
                self.weight.data[is_silent] += self.mu_silent
                if is_silent.sum() == 0:
                    break
            else : 
                break

        return out_spikes
    
    def __repr__(self):
        return f"Spiking Linear ({self.input_dim}, {self.output_dim})"


class SNN(nn.Module):
    def __init__(self, dims, **all_kwargs):
        super(SNN, self).__init__()

        layers = []
        for i, (d1, d2) in enumerate(zip(dims[:-1], dims[1:])):
            layer_kwargs = {
                k: v[i] if isinstance(v, (list, np.ndarray)) else v
                for k, v in all_kwargs.items()
            }
            layers.append(SpikingLinear(d1, d2, **layer_kwargs))

        self.outact = FirstSpikeTime.apply
        # self.outact = SpikeTime().first_spike_fn
        self.layers = RecordingSequential(*layers)

    def forward(self, input):
        out, all_spikes = self.layers(input)
        # out = self.outact(out)
        return out, all_spikes


class SNN2(nn.Module):
    def __init__(self, dims, **all_kwargs):
        super(SNN2, self).__init__()

        self.resolve_silent = all_kwargs.get('resolve_silent', True)
        layers = []
        for i, (d1, d2) in enumerate(zip(dims[:-1], dims[1:])):
            layer_kwargs = {
                k: v[i] if isinstance(v, (list, np.ndarray)) else v
                for k, v in all_kwargs.items()
            }

            layers.append(SpikingLinear2(d1, d2, **layer_kwargs))

        self.outact = SpikeTime().first_spike_fn
        self.layers = RecordingSequential(*layers)

    def forward(self, input):
        input = input.float()
        reset(self)

        out_spikes, all_spikes = self.layers(input)
        # spk_rec.append(out_spikes)
        # all_spk_rec.append(all_recs)

        # out_spikes = torch.stack(spk_rec, dim=0)
        # all_spikes = [torch.stack(all_s, dim=0) for all_s in zip(*all_spk_rec)]
        # if self.training and self.resolve_silent:
        #     is_silent = [s.sum(0).min(0)[0] == 0 for s in all_spikes]
        #     for layer, silent in zip(self.layers, is_silent):
        #         if silent.sum() != 0:
        #             layer.weight.data[silent] += layer.mu_silent
        #             # layer.weight.data[silent] += layer.weight.data[silent].abs().mean() * 0.1

        # out = self.outact(out_spikes)
        out = out_spikes
        return out, all_spikes


class SpikeCELoss(nn.Module):
    def __init__(self, xi, tau_s):
        super(SpikeCELoss, self).__init__()
        self.spike_time_fn = FirstSpikeTime.apply
        self.xi = xi
        self.tau_s = tau_s
        self.celoss = nn.CrossEntropyLoss()

    def forward(self, input, target):
        first_spikes = self.spike_time_fn(input)
        loss = self.celoss(-first_spikes / (self.xi * self.tau_s), target)
        return loss
