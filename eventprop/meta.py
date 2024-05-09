from eventprop.config import get_flat_dict_from_nested
from eventprop.models import SpikeCELoss, FirstSpikeTime
from eventprop.training import is_notebook
import torch
import argparse
from copy import deepcopy
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
import torchopt
import numpy as np
import wandb
from pathlib import Path
import glob
from PIL import Image
import shutil
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation


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
                self.model.load_state_dict(checkpoint)

        return outer_loss, {
            "meta_accs": meta_accs,
            "meta_losses": meta_losses,
            "params": adapted_params,
        }


class MAML(object):
    def __init__(self, model, flat_config) -> None:

        self.model = model
        self.flat_config = flat_config

        self.args = argparse.Namespace(**flat_config)
        self.meta_opt = torch.optim.Adam(model.parameters(), flat_config["meta_lr"])
        if flat_config.get("meta_gamma", None) is not None:
            self.meta_sch = torch.optim.lr_scheduler.ExponentialLR(
                self.meta_opt, flat_config["meta_gamma"]
            )
        else:
            self.meta_sch = None

        self.loss_fn = SpikeCELoss(**{k: flat_config[k] for k in ["alpha", "xi", "beta"]})

    def adapt(
        self, training_sample, inner_opt, n_inner_iter=None, testing_sample=None, position=None
    ):
        # Adaptation fn

        if testing_sample:
            all_test_outs = []
        if n_inner_iter is None:
            n_inner_iter = self.flat_config["num_shots"]

        if position is not None:
            pbar = tqdm(range(n_inner_iter), desc="Adaptation: ", position=position, leave=None)
        else:
            pbar = range(n_inner_iter)
        for x, y, inner_iter in zip(*training_sample, pbar):

            out, _ = self.model(x)
            loss, _, first_spikes = self.loss_fn(out, y)
            inner_opt.step(loss)
            # acc = (first_spikes.argmin(-1) == y).float().mean()
            if testing_sample is not None and inner_iter % (n_inner_iter // 100) == 0:

                *_, first_spikes = self.test(testing_sample)
                all_test_outs.append(first_spikes)

        if testing_sample is not None:
            return all_test_outs

    def test(self, testing_sample):

        if len(testing_sample[0].shape) > 3:
            testing_sample[0] = testing_sample[0].transpose(0, 1).squeeze()
        out_spikes, _ = self.model(testing_sample[0])
        loss, _, first_spikes = self.loss_fn(out_spikes, testing_sample[1])
        acc = (first_spikes.argmin(-1) == testing_sample[1]).float().mean()
        return loss, acc, first_spikes

    def get_outer_loss(self, training_batch, train=False, position=None, pbar=None):

        inner_opt = torchopt.MetaAdam(
            self.model,
            self.flat_config["lr"],
            betas=[self.flat_config["beta_1"], self.flat_config["beta_2"]],
            use_accelerated_op=True,
        )

        n_tasks = training_batch["train"][0].shape[0]

        test_losses = {"pre": [], "post": []}
        test_accs = {"pre": [], "post": []}
        outer_loss = 0

        net_state_dict = torchopt.extract_state_dict(
            self.model, by="reference", detach_buffers=True
        )
        optim_state_dict = torchopt.extract_state_dict(inner_opt, by="reference")

        if position is not None and pbar is None:
            pbar2 = tqdm(range(n_tasks), desc="Tasks", position=position, leave=None)
        else:
            pbar2 = range(n_tasks)

        for i in pbar2:

            # test before
            testing_sample = [d[i] for d in training_batch["test"]]
            test_loss, test_acc, _ = self.test(testing_sample)
            test_losses["pre"].append(test_loss.cpu().data.item())
            test_accs["pre"].append(test_acc)

            # adapt
            training_sample = [d[i] for d in training_batch["train"]]
            self.adapt(
                training_sample, inner_opt, position=position + 1 if position is not None else None
            )

            # test after
            test_loss, test_acc, _ = self.test(testing_sample)
            test_losses["post"].append(test_loss.cpu().data.item())
            test_accs["post"].append(test_acc)
            outer_loss += test_loss / n_tasks

            torchopt.recover_state_dict(self.model, net_state_dict)
            torchopt.recover_state_dict(inner_opt, optim_state_dict)

        if pbar is not None:
            # pbar.set_description(
            #     f"Test Acc : { np.mean(test_accs["pre"][-n_tasks:]):2f} -> {np.mean(test_accs["post"][-n_tasks:]):2f}"
            #     )
            pbar.set_postfix(
                {
                    "Test Acc Pre": np.mean(test_accs["pre"][-n_tasks:]),
                    "Test Acc Post": np.mean(test_accs["post"][-n_tasks:]),
                }
            )

        if train:
            self.meta_opt.zero_grad()
            outer_loss.backward()
            self.meta_opt.step()

        return outer_loss, (test_losses, test_accs)


def do_meta_training(meta_trainer, meta_train_loader, meta_test_loader, use_tqdm=True, pbar=None):

    use_wandb = wandb.run is not None

    if use_tqdm and pbar is None:
        pbar = tqdm(range(meta_trainer.flat_config["n_epochs"]), desc="Epochs", leave=None)
    else:
        pbar = range(meta_trainer.flat_config["n_epochs"])

    all_train_results = []
    all_test_results = []

    for epoch in pbar:

        if epoch > 0:

            for training_batch in tqdm(
                meta_train_loader, desc="Meta-Batches", leave=None, position=1
            ):
                train_outer_loss, train_results = meta_trainer.get_outer_loss(
                    training_batch,
                    train=True,
                    position=2,
                )
                all_train_results.append(train_results)

            if meta_trainer.meta_sch:
                meta_trainer.meta_sch.step()

        for testing_batch in meta_test_loader:
            test_outer_loss, test_results = meta_trainer.get_outer_loss(
                testing_batch, train=False, pbar=pbar
            )

            all_test_results.append(test_results)

        pbar.set_description(
            f"Test Acc : {np.mean([r[1]['pre'] for r in all_test_results[-len(meta_test_loader):]]):2f}"
            + f"-> {np.mean([r[1]['post'] for r in all_test_results[-len(meta_test_loader):]]):2f}"
        )

        if use_wandb:
            wandb.log(
                {
                    "train_loss": train_outer_loss.cpu().data.item(),
                    "test_loss": test_outer_loss.cpu().data.item(),
                    "test_acc_pre": np.mean(
                        [np.mean(r[1]["pre"]) for r in all_test_results[-len(meta_test_loader) :]]
                    ),
                    "test_acc_post": np.mean(
                        [np.mean(r[1]["post"]) for r in all_test_results[-len(meta_test_loader) :]]
                    ),
                    "train_acc_pre": np.mean(
                        [np.mean(r[1]["pre"]) for r in all_train_results[-len(meta_train_loader) :]]
                    ),
                    "train_acc_post": np.mean(
                        [
                            np.mean(r[1]["post"])
                            for r in all_train_results[-len(meta_train_loader) :]
                        ]
                    ),
                }
            )

    return meta_trainer, all_test_results, all_train_results


def create_gif(meta_trainer, train_sample, test_sample, inner_opt=None):

    if inner_opt is None:
        inner_opt = torchopt.MetaAdam(
            meta_trainer.model,
            meta_trainer.flat_config["lr"],
            betas=[meta_trainer.flat_config["beta_1"], meta_trainer.flat_config["beta_2"]],
            use_accelerated_op=True,
        )

    test_outs = meta_trainer.adapt(
        train_sample,
        inner_opt=inner_opt,
        n_inner_iter=100,
        testing_sample=test_sample,
        position=0,
    )

    path = Path("gif/imgs/")
    path.mkdir(exist_ok=True, parents=True)

    for i, test_out in enumerate(test_outs):
        plt.close()
        plt.scatter(
            test_sample[0].argmax(0)[:, 0],
            test_sample[0].argmax(0)[:, 1],
            c=F.softmin(test_out).cpu().data.numpy(),
        )
        acc = (test_out.argmin(-1) == test_sample[1].squeeze()).float().mean()
        plt.scatter(
            train_sample[0][i].argmax(0)[:, 0],
            train_sample[0][i].argmax(0)[:, 1],
            c=F.one_hot(train_sample[1][i], 3),
            marker="X",
            s=300,
        )
        plt.title(f"Acc : {acc}")
        plt.savefig("gif/imgs/{}.png".format(i))

    plt.close()
    plt.scatter(
        test_sample[0].argmax(0)[:, 0],
        test_sample[0].argmax(0)[:, 1],
        c=F.one_hot(test_sample[1].squeeze()),
    )
    plt.savefig("gif/imgs/{}.png".format(len(test_outs)))
    files = glob.glob("gif/*.png")
    files.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))

    image_array = []

    for my_file in files:

        image = Image.open(my_file)
        image_array.append(image)

    print("image_arrays shape:", np.array(image_array).shape)

    # Create the figure and axes objects
    fig, ax = plt.subplots()

    # Set the initial image
    im = ax.imshow(image_array[11], animated=True)

    def update(i):
        im.set_array(image_array[i])

    # Create the animation object
    animation_fig = animation.FuncAnimation(
        fig,
        update,
        frames=len(image_array),
        interval=200,
        blit=True,
        repeat_delay=5000,
        repeat=False,
    )

    # Show the animation
    plt.show()

    animation_fig.save("gif/animated_yy.gif")
    shutil.rmtree("gif/imgs")
