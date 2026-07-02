import os
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split

from src import utils
from . import subsample_data


@dataclass
class NotesDataset:
    X_train_branch: np.ndarray
    X_test_branch: np.ndarray
    X_train_trunk: np.ndarray
    X_test_trunk: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    latent_dim: int
    pattern_dim: int


def load_subsample_npz(dataid: int, data_root: str = "data/subsample"):
    path = os.path.join(data_root, f"subsample_dataset_seed_{dataid}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find dataset: {path}")

    data = np.load(path)
    return data["pattern"], data["efficiency"], path


def build_notes_dataset(
    dataid: int,
    latent_dim: int = 25,
    train_size: float = 0.8,
    random_state: int = 42,
    data_root: str = "data/subsample",
    use_preprocessing: bool = True,
):
    patterns, efficiencies, path = load_subsample_npz(dataid, data_root=data_root)
    y = patterns.copy()

    if use_preprocessing:
        patterns = utils.preprocessing(patterns)

    latents, pca = subsample_data.pca_transform_data(
        patterns=patterns,
        efficiencies=efficiencies,
        latent_dim=latent_dim,
    )

    y = (y + 1.0) / 2.0
    y = y.reshape(y.shape[0], -1)

    latents = latents.astype(np.float32)
    y = y.astype(np.float32)

    X_train_branch, X_test_branch, y_train, y_test = train_test_split(
        latents,
        y,
        train_size=train_size,
        random_state=random_state,
    )

    X_train_trunk = np.arange(y_train.shape[1]).reshape(-1, 1) / float(y_train.shape[1])
    X_test_trunk = X_train_trunk.copy()

    X_train_branch = X_train_branch.astype(np.float32)
    X_test_branch = X_test_branch.astype(np.float32)
    X_train_trunk = X_train_trunk.astype(np.float32)
    X_test_trunk = X_test_trunk.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)

    dataset = NotesDataset(
        X_train_branch=X_train_branch,
        X_test_branch=X_test_branch,
        X_train_trunk=X_train_trunk,
        X_test_trunk=X_test_trunk,
        y_train=y_train,
        y_test=y_test,
        latent_dim=latent_dim,
        pattern_dim=y_train.shape[1],
    )

    
    return dataset, pca, path
