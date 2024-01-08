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


def reset(net):
    for m in net.modules():
        if hasattr(m, "reset_hidden"):
            m.reset_hidden()
        # else:
        #     reset(m)


def init_weights(w, weight_scale=1.0):
    k = weight_scale * (1.0 / np.sqrt(w.shape[1]))
    nn.init.uniform_(w, -k, k)
    print(k)


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
            elif isinstance(x, dict):
                x = x["output"]

        return {"output": x, "recordings": recs}


class SpikingLinear_ev(nn.Module):
    def __init__(self, d1, d2, **kwargs):
        super(SpikingLinear_ev, self).__init__()

        self.input_dim = d1
        self.output_dim = d2
        self.dt = kwargs.get("dt", 1e-3)
        self.tau_m = kwargs.get("tau_m", 20e-3)
        self.tau_s = kwargs.get("tau_s", 5e-3)
        self.resolve_silent = kwargs.get("resolve_silent", False)

        self.mu = kwargs.get("mu", 0.1)
        self.sigma = kwargs.get("sigma", 0.1)
        self.scale = kwargs.get("scale", 1.0)
        self.mu_silent = 1 / np.sqrt(d1)

        self.dropout_p = kwargs.get("dropout", None)

        # self.alpha = np.exp(-self.dt / self.tau_s)
        # self.beta = np.exp(-self.dt / self.tau_m)

        self.alpha = 1 - self.dt / self.tau_s
        self.beta = 1 - self.dt / self.tau_m

        self.weight = nn.Parameter(torch.Tensor(d2, d1))

        self.init_mode = kwargs.get("init_mode", "kaiming")
        if self.init_mode == "kaiming":
            nn.init.kaiming_normal_(self.weight)
        elif self.init_mode == "kaiming_both":
            mu = 1 / np.sqrt(nn.init._calculate_fan_in_and_fan_out(self.weight)[0])
            nn.init.normal_(self.weight, mu, mu)
        elif self.init_mode == "normal":
            nn.init.normal_(self.weight, self.mu, self.sigma)
        elif self.init_mode == "uniform":
            nn.init.uniform_(self.weight, -self.mu, self.mu)
        else:
            raise ValueError(f"Invalid init_mode {self.init_mode}")

        self.weight.data *= self.scale

        self.reset_to_zero = kwargs.get("reset_to_zero", False)

        self.forward = lambda input: self.EventProp.apply(
            input, self.weight, self.manual_forward, self.manual_backward
        )

    @staticmethod
    class EventProp(Function):
        @staticmethod
        def forward(ctx, input, weights, manual_forward, manual_backward):
            ctx.backward = manual_backward
            pack, output = manual_forward(input)
            ctx.save_for_backward(*pack)
            return output

        @staticmethod
        def backward(ctx, *grad_output):
            backward = ctx.backward
            pack = ctx.saved_tensors
            grad_input, grad_weights, *_ = backward(grad_output[0], pack)
            return grad_input, grad_weights, None, None

    def __repr__(self):
        return super().__repr__()[:-1] + f"{self.input_dim}, {self.output_dim})"

    def manual_forward(self, input):
        steps = input.shape[0]

        V = torch.zeros(steps, input.shape[1], self.output_dim).to(device)
        I = torch.zeros(steps, input.shape[1], self.output_dim).to(device)
        V_spikes = torch.zeros(steps, input.shape[1], self.output_dim).to(device)
        output = torch.zeros(steps, input.shape[1], self.output_dim).to(device)

        # input = torch.roll(input, 1, dims=0)
        # print("Spike is at time", input.argmax().data.item())

        while True:
            for i in range(1, steps):
                # spikes = (V[i - 1] > 1.0).float()
                # V[i - 1] = (1 - spikes) * V[i - 1]

                input_t = F.linear(input[i - 1].float(), self.weight)
                if self.dropout_p:
                    input_t = F.dropout(input_t, p=self.dropout_p)
                I[i] = self.alpha * I[i - 1] + input_t
                V[i] = self.beta * V[i - 1] + (1 - self.beta) * I[i]

                V_spikes[i] = V[i]

                spikes = (V[i] > 1.0).float()
                V[i] = (1 - spikes) * V[i]

                # if self.reset_to_zero:
                #     V[i] = (1 - spikes) * V[i]
                # else:
                #     V[i] -= spikes

                output[i] = spikes

            if self.training and self.resolve_silent:
                is_silent = output.sum(0).mean(0) == 0
                self.weight.data[is_silent] += self.mu_silent
                if is_silent.sum() == 0:
                    break
            else:
                break

        # return (input, V, V_spikes, I, output), (output, V)
        return (
            (input, V, V_spikes, I, output),
            {
                "output": output,
                "V": V,
                "I": I,
            },
        )

    def manual_backward(self, grad_output, pack):
        input, _, V, I, post_spikes = pack
        # input = torch.roll(input, 1, dims=0)
        # post_spikes = torch.roll(post_spikes, 1, dims=0)
        steps = input.shape[0]

        lV = torch.zeros(steps, input.shape[1], self.output_dim).to(device)
        lI = torch.zeros(steps, input.shape[1], self.output_dim).to(device)

        grad_input = torch.zeros(steps, input.shape[1], input.shape[2]).to(device)
        grad_weight = torch.zeros(input.shape[1], *self.weight.shape).to(device)
        jumps, V_dots = [], []

        for i in range(steps - 2, -1, -1):
            delta = lV[i + 1] - lI[i + 1]
            grad_input[i] = F.linear(delta, self.weight.t())

            # Euler
            lI[i] = self.alpha * lI[i + 1] + (1 - self.alpha) * lV[i + 1]
            lV[i] = self.beta * lV[i + 1]

            # Jump
            V_dot = I[i + 1] - V[i + 1] + 1e-10
            V_dots.append(V_dot)
            jump = post_spikes[i + 1] * ((lV[i + 1] + grad_output[i + 1]) / V_dot)
            jumps.append(jump)

            if jump.mean().data.item() != 0:
                to_print = {
                    "jump": jump.data,
                    "grad_output": grad_output[i + 1].data,
                    "grad_input": grad_input[i].data,
                    "V_dot": (I[i] - V[i]).data,
                    "lV[i+1]": lV[i + 1].data,
                    "lI[i+1]": lI[i + 1].data,
                    "lV[i]": lV[i].data,
                    "lI[i]": lI[i].data,
                    # "I[i]": I[i].data,
                    # "I[i-1]": I[i - 1].data,
                    # "I[i+1]": I[i + 1].data,
                    # "V[i]": V[i].data,
                    # "V[i-1]": V[i - 1].data,
                    # "V[i+1]": V[i + 1].data,
                }
                # print(
                #     str(
                #         f"Got spike at time {i}, jump is {jump.cpu().data.numpy()}"
                #         + f"V_dot is {V_dot.cpu().data.numpy()}"
                #         + f"error is {grad_output[i + 1].cpu().data.numpy() }"
                #     )
                # )
                # print(to_print)

            lV[i] += jump

            # Accumulate grad
            spike_bool = input[i].float()
            grad_weight -= self.tau_s * spike_bool.unsqueeze(1) * lI[i].unsqueeze(2)

        return grad_input, grad_weight, lV, lI


class SpikingLinear_su(nn.Module):
    def __init__(self, d1, d2, **kwargs) -> None:
        super().__init__()
        self.input_dim, self.output_dim = d1, d2
        self.mu = kwargs.get("mu", 1.0)
        self.mu_silent = 1 / np.sqrt(d1)
        self.resolve_silent = kwargs.get("resolve_silent", False)
        self.input_dropout_p = kwargs.get("input_dropout", None)
        if self.input_dropout_p:
            self.input_dropout = nn.Dropout(p=self.input_dropout_p)

        self.weight = nn.Parameter(torch.Tensor(d2, d1))
        nn.init.kaiming_normal_(self.weight)
        self.weight.data *= self.mu

        self.syn = snn.Synaptic(
            alpha=np.exp(-kwargs["dt"] / kwargs["tau_s"]),
            beta=np.exp(-kwargs["dt"] / kwargs["tau_m"]),
            init_hidden=True,
            reset_mechanism="zero",
            threshold=1.0,
        )

    def forward(self, input):
        if self.input_dropout_p:
            input = self.input_dropout(input)
        while True:
            out_spikes = []
            voltages = []
            currents = []
            for t, in_spikes in enumerate(input):
                out = F.linear(in_spikes, self.weight)
                spikes = self.syn(out)
                out_spikes.append(spikes)
                voltages.append(self.syn.mem)
                currents.append(self.syn.syn)

            out_spikes = torch.stack(out_spikes, dim=0)
            voltages = torch.stack(voltages, dim=0)
            currents = torch.stack(currents, dim=0)

            if self.training and self.resolve_silent:
                is_silent = out_spikes.sum(0).mean(0) == 0
                self.weight.data[is_silent] += self.mu_silent
                if is_silent.sum() == 0:
                    break
            else:
                break

        return {"output": out_spikes, "V": voltages, "I": currents}

    def __repr__(self):
        return f"Spiking Linear ({self.input_dim}, {self.output_dim})"


layer_types = {str(t): t for t in [SpikingLinear_ev, SpikingLinear_su]}
model_types = {
    m: t
    for m, t in zip(["eventprop", "snntorch"], [SpikingLinear_ev, SpikingLinear_su])
}


class SNN(nn.Module):
    def __init__(self, dims, **all_kwargs):
        super(SNN, self).__init__()

        self.get_first_spikes = all_kwargs.get("get_first_spikes", False)

        self.layer_type = all_kwargs.get("layer_type", None)
        self.model_type = all_kwargs.get("model_type", None)

        assert not (
            self.layer_type is None and self.model_type is None
        ), "Must specify layer_type or model_type"

        if self.model_type is not None:
            assert (
                self.model_type in model_types
            ), f"Invalid model_type {self.model_type}"
            layer = model_types[self.model_type]
            self.eventprop = self.model_type == "eventprop"
        else:
            assert (
                self.layer_type in layer_types
            ), f"Invalid layer_type {self.layer_type}"
            layer = layer_types[self.layer_type]
            self.eventprop = self.layer_type == str(SpikingLinear_ev)

        layers = []
        for i, (d1, d2) in enumerate(zip(dims[:-1], dims[1:])):
            layer_kwargs = {
                k: v[i] if isinstance(v, (list, np.ndarray)) else v
                for k, v in all_kwargs.items()
            }
            if i != 0:
                layer_kwargs["dropout"] = None

            layers.append(layer(d1, d2, **layer_kwargs))

        if self.get_first_spikes:
            self.outact = SpikeTime().first_spike_fn

        self.layers = RecordingSequential(*layers)

    def forward(self, input):
        if not self.eventprop:
            input = input.float()
            reset(self)
        # out, all_recs = self.layers(input)
        out_dict = self.layers(input)
        out = out_dict["output"]
        if self.get_first_spikes:
            out = self.outact(out)
        return out, out_dict["recordings"]


class SpikeCELoss(nn.Module):
    def __init__(self, xi):
        super(SpikeCELoss, self).__init__()
        self.spike_time_fn = FirstSpikeTime.apply
        # self.spike_time_fn = SpikeTime().first_spike_fn
        self.xi = xi
        self.celoss = nn.CrossEntropyLoss()

    def forward(self, input, target):
        first_spikes = self.spike_time_fn(input)
        loss = self.celoss(-first_spikes / (self.xi), target)
        return loss


class FirstSpikeTime(Function):
    @staticmethod
    def forward(ctx, input):
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
        k = (
            F.one_hot(first_spike_times.long(), input.shape[0]).float().permute(2, 0, 1)
        )  # T x B x N
        grad_input = k * grad_output.unsqueeze(0)
        return grad_input
