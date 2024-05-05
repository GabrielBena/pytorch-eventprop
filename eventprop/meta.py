from eventprop.config import get_flat_dict_from_nested
from eventprop.models import SpikeCELoss, FirstSpikeTime
from eventprop.training import is_notebook
import torch
import argparse
from copy import deepcopy
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook


class REPTILE(object):

    def __init__(self, model, default_config) -> None:

        self.model = model
        self.default_config = default_config
        self.flat_config = get_flat_dict_from_nested(default_config)
        self.args = argparse.Namespace(**self.flat_config)

        self.meta_optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.flat_config["meta-lr"],
            weight_decay=self.flat_config.get("meta-weight_decay", 0),
        )

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

    def adapt(self, meta_sample, use_tqdm=False, position=0):

        inputs, targets = meta_sample
        inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

        # start optim from scratch
        if getattr(self, "inner_optimizer", None) is None:
            inner_optimizer = torch.optim.Adam(
                dict(self.model.meta_named_parameters()).values(),
                lr=self.flat_config["inner-lr"],
                weight_decay=self.flat_config.get("weight_decay", 0),
            )
        # use the same optimizer
        else:
            inner_optimizer = self.inner_optimizer

        n_shots = (
            inputs.size(0)
            if self.flat_config["num_shots"] is None
            else self.flat_config["num_shots"]
        )

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
            out_spikes, recordings = self.model(input)
            loss = self.inner_loss_fn(out_spikes, target)[0]
            inner_optimizer.zero_grad()
            loss.backward()
            inner_optimizer.step()

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

        adapted_params = {}

        self.model.reset_recordings()

        for task, train_meta_sample in enumerate(zip(*meta_batch["train"])):

            # create checkpoint
            start_weights = deepcopy(self.model.state_dict())

            # test before adapt
            test_meta_sample = [s[task] for s in meta_batch["test"]]
            pre_acc, pre_loss, _ = self.inner_test(test_meta_sample, params=start_weights)
            pre_acc = pre_acc.detach().cpu().numpy()
            meta_accs["pre"].append(pre_acc)
            meta_losses["pre"].append(pre_loss)

            # adapt
            self.adapt(
                train_meta_sample,
                use_tqdm=use_tqdm and train and not is_notebook() and False,
                position=position + 1,
            )
            # new weights after adaptation
            candidate_weights = self.model.state_dict()

            # test after adapt, using the adapted weights
            post_acc, post_loss, _ = self.inner_test(test_meta_sample, params=candidate_weights)
            post_acc = post_acc.detach().cpu().numpy()
            meta_accs["post"].append(post_acc)
            meta_losses["post"].append(post_loss)
            outer_loss += post_loss

            if train:

                # annealed learning rate
                alpha = self.flat_config["meta-lr"] * (1 - self.outer_iter / self.total_outer_iter)

                # update the model weights
                updated_params = {
                    candidate: (
                        start_weights[candidate]
                        + alpha * (candidate_weights[candidate] - start_weights[candidate])
                    )
                    for candidate in candidate_weights
                }
                self.model.load_state_dict(updated_params)
                self.outer_iter += 1
            else:
                # return to checkpoint
                self.model.load_state_dict(start_weights)
        return outer_loss, {
            "meta_accs": meta_accs,
            "meta_losses": meta_losses,
            "params": adapted_params,
        }
