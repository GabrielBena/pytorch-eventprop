import matplotlib.pyplot as plt
import numpy as np

import torch
from torchvision import datasets, transforms

import tonic
from yingyang.dataset import YinYangDataset

from new_models import SNN, SNN2, SpikeCELoss
from training import train, test, encode_data
import yaml
import random
import snntorch as snn
import os

import torch.nn as nn
from new_models import SpikingLinear, SpikingLinear2
import math
import torch.distributions as dists


def get_lif_kernel(tau_mem=20e-3, tau_syn=10e-3, dt=1e-3):
    """Computes the linear filter kernel of a simple LIF neuron with exponential current-based synapses.

    Args:
        tau_mem: The membrane time constant
        tau_syn: The synaptic time constant
        dt: The timestep size

    Returns:
        Array of length 10x of the longest time constant containing the filter kernel

    """
    tau_max = np.max((tau_mem, tau_syn))
    ts = np.arange(0, int(tau_max * 10 / dt)) * dt
    n = len(ts)
    kernel = np.empty(n)
    I = 1.0  # Initialize current variable for single spike input
    U = 0.0
    dcy1 = np.exp(-dt / tau_mem)
    dcy2 = np.exp(-dt / tau_syn)
    for i, t in enumerate(ts):
        kernel[i] = U
        U = dcy1 * U + (1.0 - dcy1) * I
        I *= dcy2
    return kernel


def _get_epsilon(calc_mode, tau_mem, tau_syn, timestep=1e-3):
    if calc_mode == "analytical":
        return _epsilon_analytical(tau_mem, tau_syn)

    elif calc_mode == "numerical":
        return _epsilon_numerical(tau_mem, tau_syn, timestep)

    else:
        raise ValueError("invalid calc mode for epsilon")


def _epsilon_analytical(tau_mem, tau_syn):
    epsilon_bar = tau_syn
    epsilon_hat = (tau_syn**2) / (2 * (tau_syn + tau_mem))

    return epsilon_bar, epsilon_hat


def _epsilon_numerical(tau_mem, tau_syn, timestep):
    kernel = get_lif_kernel(tau_mem, tau_syn, timestep)
    epsilon_bar = kernel.sum() * timestep
    epsilon_hat = (kernel**2).sum() * timestep

    return epsilon_bar, epsilon_hat


class Initializer:
    """
    Abstract Base Class for Initializer Objects.
    Will fork for any Sequential
    """

    def __init__(
        self, scaling="1/sqrt(k)", sparseness=1.0, bias_scale=1.0, bias_mean=0.0
    ):
        self.scaling = scaling
        self.sparseness = sparseness
        self.bias_scale = bias_scale
        self.bias_mean = bias_mean

    def initialize(self, model):
        for target in model.layers : 
            if isinstance(target, (SpikingLinear, SpikingLinear2, nn.Linear)):
                self.initialize_connection(target)

            elif isinstance(target, snn.SpikingNeuron):
                pass

            else:
                raise TypeError(
                    "Target object is unsupported. must be nn.Linear or Layer instance."
                )

    def initialize_layer(self, layer):
        """
        Initializes all connections in a `Layer` object
        """
        self.initialize_connection(layer)

    def initialize_connection(self, connection):
        """
        Initializes weights of a `Connection` object
        """
        raise NotImplementedError

    def _apply_scaling(self, weights):
        """
        Implements weight scaling options
        """
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weights)

        if self.scaling is None:
            return weights

        elif self.scaling == "1/sqrt(k)":
            return weights / math.sqrt(fan_in)

        elif self.scaling == "1/k":
            return weights / fan_in

        elif isinstance(self.scaling, float):
            return weights * self.scaling

        else:
            return weights

    def _apply_sparseness(self, weights):
        """
        Applies sparse mask to weight matrix
        """
        if self.sparseness < 1.0:
            x = torch.rand_like(weights)
            mask = x.le(self.sparseness)

            # apply mask & correct weights for sparseness
            weights.mul_(1.0 / math.sqrt(self.sparseness) * mask)

        return weights

    def _set_weights_and_bias(self, connection, weights=None):
        """
        Set weights and biases of a connection object
        """
        # set weights
        self._set_weights(connection, weights)

        # set biases if used
        if hasattr(connection, 'bias') : 
            self._set_biases(connection)

        # apply constraints
        if hasattr(connection, 'apply_constraints') :
            connection.apply_constraints()

    def _set_weights(self, connection, weights):
        """
        Sets weights at connection object and applies scaling and sparseness
        """
        # apply scaling
        weights = self._apply_scaling(weights)

        # apply sparseness
        weights = self._apply_sparseness(weights)

        # set weights
        with torch.no_grad():
            connection.weight.data = weights

    def _set_biases(self, connection):
        """
        Biases are always initialized from a uniform distribution and scaled by 1/sqrt(k)
        """
        if connection.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(
                connection.op.weight
            )
            bound = self.bias_scale / np.sqrt(fan_in)

            with torch.no_grad():
                connection.bias.uniform_(
                    -bound + self.bias_mean, bound + self.bias_mean
                )

    def _get_weights(self, *params):
        raise NotImplementedError


class FluctuationDrivenNormalInitializer(Initializer):
    """
    Implements flucutation-driven initialization as described in:
    Rossbroich, J., Gygax, J. & Zenke, F. Fluctuation-driven initialization for spiking neural network training. arXiv [cs.NE] (2022)

    params:

        Initialization parameters
        mu_U:      The target membrane potential mean
        xi:        The target distance between mean membrane potential and firing threshold in units of standard deviation

        Network and input parameters
        nu:         Estimated presynaptic firing rate (currently global for the whole network)
        timestep:   Simulation timestep (needed for calculation of epsilon numerically)

    The initialization method can additionally be parameterized through

    epsilon_calc_mode:          [set to `numerical` or `analytical`]
                                Analytical or numerical calculation of epsilon kernels (numerically is usually
                                better for large timesteps >=1ms)

    alpha:                      [set to float between 0 and 1]
                                Scales weights so that this proportion of membrane potential fluctuations
                                (variance) is accounted for by feed-forward connections.
    """

    def __init__(
        self, mu_u, xi, nu, timestep, epsilon_calc_mode="numerical", alpha=0.9, **kwargs
    ):
        super().__init__(
            scaling=None,  # None, as scaling is implemented in the weight sampling
            **kwargs
        )

        self.mu_u = mu_u
        self.xi = xi
        self.nu = nu
        self.timestep = timestep
        self.epsilon_calc_mode = epsilon_calc_mode
        self.alpha = alpha

    def _calc_epsilon(self, neuron_model):
        """
        Calculates epsilon_bar and epsilon_hat, the integrals of the PSP kernel from a target
        neuron group `dst`
        """
        if not hasattr(neuron_model, "tau_m"):
            assert hasattr(neuron_model, 'beta')
            tau_m = 1 / (1 - neuron_model.beta)
        
            assert hasattr(neuron_model, 'alpha')
            tau_s = 1 / (1 - neuron_model.alpha)
        
        else  :
            tau_m = neuron_model.tau_m
            tau_s = neuron_model.tau_s

        ebar, ehat = _get_epsilon(
            self.epsilon_calc_mode,
            tau_m *1e-3,
            tau_s *1e-3,
            self.timestep,
        )

        return ebar, ehat

    def _get_weights(self, connection, mu_w, sigma_w):
        shape = connection.weight.shape

        # sample weights
        weights = dists.Normal(mu_w, sigma_w).sample(shape)

        return weights

    def _get_weight_parameters_con(self, connection):
        """
        Calculates weight parameters for a single connection
        """

        theta = 1.0  # Theta (firing threshold) is hardcoded as in the LIFGroup class

        # Read out relevant attributes from connection object
        n, _ = torch.nn.init._calculate_fan_in_and_fan_out(connection.weight)
        if hasattr(connection, 'syn') : 
            ebar, ehat = self._calc_epsilon(connection.syn)
        else :
            ebar, ehat = self._calc_epsilon(connection)

        mu_w = self.mu_u / (n * self.nu * ebar)
        sigma_w = math.sqrt(
            1 / (n * self.nu * ehat) * ((theta - self.mu_u) / self.xi) ** 2 - mu_w**2
        )

        return mu_w, sigma_w

    def initialize_connection(self, connection):
        """
        Initializes weights of a single `Connection` object
        """
        # get parameters
        mu_w, sigma_w = self._get_weight_parameters_con(connection)
        # get weights
        weights = self._get_weights(connection, mu_w, sigma_w)
        # set weights
        self._set_weights_and_bias(connection, weights)

    def _get_weight_parameters_dst(self, dst):
        """
        Calculates weight parameters for all connections targeting a
        neuron group `dst`
        """

        theta = 1.0  # Theta (firing threshold) is hardcoded as in the LIFGroup code

        ebar, ehat = self._calc_epsilon(dst)

        # Read out some properties of the afferent connections
        nb_recurrent = len([c for c in dst.afferents if c.is_recurrent])
        nb_ff = len(dst.afferents) - nb_recurrent

        if nb_recurrent >= 1:
            # If there is at least one recurrent connection, use alpha to scale the
            # contribution to the membrane potential fluctuations
            alpha = self.alpha
        else:
            # Otherwise alpha equals one (all contribution to fluctuations spread across feed-forward connections)
            alpha = 1.0

        # Sum of all inputs
        N_total = int(
            sum(
                [
                    torch.nn.init._calculate_fan_in_and_fan_out(c.op.weight)[0]
                    for c in dst.afferents
                ]
            )
        )

        # List with weight parameters for each connection
        params = []

        for c in dst.afferents:
            # Number of presynaptic neurons
            N, _ = torch.nn.init._calculate_fan_in_and_fan_out(c.op.weight)

            # mu_w is the same for all connections
            mu_w = self.mu_u / (N_total * self.nu * ebar)

            # Compute sigma scaling factor
            if c.is_recurrent:
                scale = (1 - alpha) / nb_recurrent
            else:
                scale = alpha / nb_ff

            # sigma_w for this connection
            sigma_w = math.sqrt(
                scale / (N * self.nu * ehat) * ((theta - self.mu_u) / self.xi) ** 2
                - mu_w**2
            )

            # append to parameter list
            params.append((mu_w, sigma_w))

        return params

    def _initialize_layer(self, layer):
        # Loop through each population in this layer
        for neurons in layer.neurons:
            # Consider all afferents to this population
            # and compute weight parameters for each connection
            weight_params = self._get_weight_parameters_dst(neurons)

            # Initialize each connection
            for idx, connection in enumerate(neurons.afferents):
                # Read out parameters for weight distribution
                mu_w, sigma_w = weight_params[idx]
                # sample weights
                weights = self._get_weights(connection, mu_w, sigma_w)
                # set weights
                self._set_weights_and_bias(connection, weights)


class FluctuationDrivenCenteredNormalInitializer(FluctuationDrivenNormalInitializer):
    """
    Simpler version of the FluctuationDrivenNormalInitializer class.
    Here, the normal distribution is centered, so that initialization of synaptic weights can be
    achieved by setting a target membrane potential standard deviation sigma_u = 1/xi
    """

    def __init__(
        self, sigma_u, nu, timestep, epsilon_calc_mode="numerical", alpha=0.9, **kwargs
    ):
        super().__init__(
            mu_u=0.0,
            xi=1 / sigma_u,
            nu=nu,
            timestep=timestep,
            epsilon_calc_mode=epsilon_calc_mode,
            alpha=alpha,
            **kwargs
        )

if __name__ == "__main__":


    model_kwars = {
        "T": 20,
        "dt": 1,
        "tau_m": 20,
        "tau_s": 1,
        "mu": 1,
        "resolve_silent": True,
    }

    dims = [784, 100, 10]
    snntorch_model = SNN2(dims, **model_kwars)
   
    dt, T = model_kwars['dt'], model_kwars['T']
    sigma_nu, nu = 1, 15

    initializer = FluctuationDrivenCenteredNormalInitializer(
        sigma_u=sigma_nu, nu=nu, timestep=dt
    )

    initializer.initialize(snntorch_model)


