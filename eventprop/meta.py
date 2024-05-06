from eventprop.config import get_flat_dict_from_nested
from eventprop.models import SpikeCELoss, FirstSpikeTime
from eventprop.training import is_notebook
import torch
import argparse
from copy import deepcopy
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
import torchopt


class REPTILE(object):

    def __init__(self, model, flat_config) -> None:

        self.model = model
        self.flat_config = flat_config
        self.args = argparse.Namespace(**self.flat_config)

        self.inner_loss_fn = SpikeCELoss(
            alpha=self.flat_config["alpha"],
            xi=self.flat_config["xi"],
            beta=self.flat_config["beta"],
        )
        self.first_spk_fn = FirstSpikeTime.apply

        self.outer_iter = 0
        self.total_outer_iter = (
            self.flat_config["n_tasks_per_split_train"] * self.flat_config["n_epochs"]
        )

    def adapt(self, meta_sample, use_tqdm=False, position=0, shuffle=True, inner_params=None):

        inputs, targets = meta_sample
        inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

        if inner_params is None:
            inner_params = dict(self.model.named_parameters())

        # start optim from scratch
        if getattr(self, "inner_optimizer", None) is None:
            inner_optimizer = torchopt.Adam(
                inner_params.values(),
                lr=self.flat_config["lr"],
                weight_decay=self.flat_config.get("weight_decay", 0),
                betas=(
                    self.flat_config.get("beta_1", 0.9),
                    self.flat_config.get("beta_2", 0.999),
                ),
            )
        # use the same optimizer
        else:
            inner_optimizer = self.inner_optimizer

        n_shots = (
            inputs.size(0)
            if self.flat_config["num_shots"] is None
            else self.flat_config["num_shots"]
        )

        if shuffle:
            shuffled_idxs = torch.randperm(inputs.size(0))
            inputs = inputs[shuffled_idxs]
            targets = targets[shuffled_idxs]

        if use_tqdm:
            if is_notebook():
                pbar = tqdm_notebook(
                    range(n_shots), desc="Adaptation: ", position=position, leave=None
                )
            else:
                pbar = tqdm(range(n_shots), desc="Adaptation: ", position=position, leave=None)
        else:
            pbar = range(n_shots)

        # batch one
        for input, target, _ in zip(
            inputs,
            targets,
            pbar,
        ):
            out_spikes, recordings = self.model(input, params=inner_params)
            loss = self.inner_loss_fn(out_spikes, target)[0]
            inner_optimizer.zero_grad()
            loss.backward()
            inner_optimizer.step()

        return inner_params

    def inner_test(self, meta_sample, params=None):

        inputs, targets = meta_sample
        inputs = inputs.transpose(0, 1).to(self.args.device).squeeze()
        out_spikes, recordings = self.model(inputs, params=params)
        first_spikes = self.first_spk_fn(out_spikes)
        acc = (first_spikes.argmin(-1) == targets).float().mean()
        loss = self.inner_loss_fn(out_spikes, targets)[0]
        return acc, loss, recordings

    def get_outer_loss(self, meta_batch, train=False, use_tqdm=False, position=0):

        meta_losses = {"pre": [], "post": []}
        meta_accs = {"pre": [], "post": []}

        outer_loss = 0

        adapted_params = []

        for task, train_meta_sample in enumerate(zip(*meta_batch["train"])):

            # create checkpoint
            checkpoint = deepcopy(dict(self.model.named_parameters()))

            # test before adapt
            test_meta_sample = [s[task] for s in meta_batch["test"]]
            pre_acc, pre_loss, _ = self.inner_test(test_meta_sample, params=checkpoint)
            pre_acc = pre_acc.detach().cpu().numpy()
            meta_accs["pre"].append(pre_acc)
            meta_losses["pre"].append(pre_loss)

            # adapt
            candidate_weights = self.adapt(
                train_meta_sample,
                use_tqdm=use_tqdm and train and not is_notebook() and False,
                position=position + 1,
                inner_params=None,
            )

            # test after adapt, using the adapted weights
            post_acc, post_loss, _ = self.inner_test(test_meta_sample, params=candidate_weights)
            post_acc = post_acc.detach().cpu().numpy()
            meta_accs["post"].append(post_acc)
            meta_losses["post"].append(post_loss)
            outer_loss += post_loss

            if train:

                if self.flat_config["annealing"] == "linear":
                    # annealed learning rate
                    alpha = self.flat_config["step_size"] * (
                        1 - self.outer_iter / self.total_outer_iter
                    )
                    print(alpha)
                else:
                    raise NotImplementedError("Annealing strategy not implemented")

                # update the model weights
                updated_params = {
                    candidate: (
                        checkpoint[candidate]
                        + alpha * (candidate_weights[candidate] - checkpoint[candidate])
                    )
                    for candidate in candidate_weights
                }
                self.model.load_state_dict(updated_params)
                self.outer_iter += 1
                adapted_params.append(candidate_weights)

            else:
                # return to checkpoint
                self.model.load_state_dict(start_weights)

        return outer_loss, {
            "meta_accs": meta_accs,
            "meta_losses": meta_losses,
            "params": adapted_params,
        }
