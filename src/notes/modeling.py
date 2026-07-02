import os
from dataclasses import dataclass

import numpy as np
import torch
import deepxde as dde

from src import utils


@dataclass
class NotesModelConfig:
    latent_dim: int = 25
    trunk_dim: int = 1
    hidden_width: int = 60
    hidden_depth: int = 3
    activation: str = "relu"
    initializer: str = "Glorot normal"
    scale_factor: float = 1.0
    bias_factor: float = 0.5366
    output_activation: str = "sigmoid"


def set_all_seeds(seed: int):
    dde.config.set_random_seed(seed)
    utils.set_seed(seed)


def build_deeponet_data(dataset):

    return dde.data.TripleCartesianProd(
        X_train=(dataset.X_train_branch, dataset.X_train_trunk),
        y_train=dataset.y_train,
        X_test=(dataset.X_test_branch, dataset.X_test_trunk),
        y_test=dataset.y_test,
    )


def build_notes_net(config: NotesModelConfig):
    branch_layers = [config.latent_dim] + [config.hidden_width] * config.hidden_depth
    trunk_layers = [config.trunk_dim] + [config.hidden_width] * config.hidden_depth

    return utils.DeepONetCartesianProdPlus(
        branch_layers,
        trunk_layers,
        config.activation,
        config.initializer,
        scale_factor=config.scale_factor,
        bias_factor=config.bias_factor,
        output_activation=config.output_activation,
    )


def build_notes_model(dataset, config: NotesModelConfig):
    dde_data = build_deeponet_data(dataset)
    net = build_notes_net(config)
    return dde.Model(dde_data, net)


class BinaryBCELossWithBinarization:
    def __init__(self, alpha: float = 20.0):
        self.alpha = alpha
        self.loss_fn = utils.binary_cross_entropy

    def __call__(self, y_true, y_pred):
        bce = self.loss_fn(y_true, y_pred)
        reg = self.alpha * y_pred * (1.0 - y_pred)
        return bce + reg.mean()


def binary_cross_entropy_np(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1.0 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))


def train_notes_model(
    model,
    seed: int,
    output_dir: str = "model/subsample",
    lr: float = 1e-3,
    alpha: float = 20.0,
    iterations_per_stage: int = 40000,
    num_stages: int = 10,
):
    os.makedirs(output_dir, exist_ok=True)
    save_prefix = os.path.join(output_dir, f"model{seed}", f"model{seed}.ckpt")
    os.makedirs(os.path.dirname(save_prefix), exist_ok=True)

    loss_fn = BinaryBCELossWithBinarization(alpha=alpha)
    model.compile(
        "adam",
        lr=lr,
        loss=loss_fn,
        metrics=[binary_cross_entropy_np],
    )

    last_losshistory = None
    last_train_state = None

    for stage in range(num_stages):
        print(f"Training stage {stage + 1}/{num_stages}")
        last_losshistory, last_train_state = model.train(
            iterations=iterations_per_stage,
            model_save_path=save_prefix,
        )

        if hasattr(model.net, "scale_factor"):
            model.net.scale_factor *= 2
            print(f"Updated scale_factor to {model.net.scale_factor}")

    history_path = os.path.join(output_dir, f"model{seed}_losshistory.npz")
    np.savez(
        history_path,
        train_loss=last_losshistory.loss_train,
        val_loss=last_losshistory.loss_test,
        val_metric=last_losshistory.metrics_test,
    )
    print(f"Saved loss history to {history_path}")

    return last_losshistory, last_train_state


def load_trained_notes_model(dataset, config: NotesModelConfig, checkpoint_path: str, device=None):
    model = build_notes_model(dataset, config)
    model.compile(optimizer="adam", lr=1e-3)
    model.restore(save_path=checkpoint_path, verbose=1)

    for param in model.net.parameters():
        param.requires_grad = False
    model.net.eval()

    if device is not None:
        model.net.to(device)

    return model
