import os
import numpy as np
from sklearn.decomposition import PCA


def npz_filter_data(
    data,
    efficiency,
    wavelength_range=(0.0, 1400.0),
    angle_range=(0.0, 90.0),
):
    mask = (
        (data["efficiency"] >= efficiency)
        & (data["wavelength"] >= wavelength_range[0])
        & (data["wavelength"] <= wavelength_range[1])
        & (data["angle"] >= angle_range[0])
        & (data["angle"] <= angle_range[1])
    )

    result = {}
    for key in data.files:
        result[key] = data[key][mask]

    return result


def subsample_data(
    path,
    efficiency,
    seed,
    prob,
    wavelength_range=(0.0, 1400.0),
    angle_range=(0.0, 90.0),
):
    dataset = np.load(path)
    dataset = npz_filter_data(
        dataset,
        efficiency=efficiency,
        wavelength_range=wavelength_range,
        angle_range=angle_range,
    )

    rng = np.random.default_rng(seed)
    n = dataset["pattern"].shape[0]
    mask = rng.random(n) < prob

    info = {
        "seed": seed,
        "probability": prob,
        "num_original": n,
        "num_subsampled": int(mask.sum()),
        "efficiency_threshold": efficiency,
    }

    return (dataset["pattern"][mask], dataset["efficiency"][mask]), info


def save_data(data, path, seed, info=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    save_dict = {
        "pattern": data[0],
        "efficiency": data[1],
        "seed": seed,
    }

    if info is not None:
        save_dict.update(info)

    np.savez(path, **save_dict)
    print(f"Saved data to {path}")


def pca_transform_data(patterns, efficiencies, latent_dim=25, save_pca_path=None, preprocessing=False):
    patterns = patterns.reshape(patterns.shape[0], -1)

    pca = PCA(n_components=latent_dim)
    latent = pca.fit_transform(patterns)

    if save_pca_path is not None:
        os.makedirs(os.path.dirname(save_pca_path), exist_ok=True)
        np.savez(
            save_pca_path,
            components=pca.components_,
            mean=pca.mean_,
            explained_variance_ratio=pca.explained_variance_ratio_,
        )
        print(f"Saved PCA decoder to {save_pca_path}")

    return latent, pca

if __name__ == "__main__":
    data_path = "../../data/dataset.npz"
    output_dir = "../../data/subsample"
    pca_decoder_dir = "../../model/subsample_pca"

    efficiency = 0.9
    prob = 0.025
    latent_dim = 25
    seeds = np.random.randint(0, 1000, size=100)

    for seed in seeds:
        print(f"Running subsampling with seed = {seed}")

        data, info = subsample_data(
            path=data_path,
            efficiency=efficiency,
            seed=int(seed),
            prob=prob,
        )

        output_data_path = f"{output_dir}/subsample_dataset_seed_{seed}.npz"
        pca_decoder_path = f"{pca_decoder_dir}/subsample_pca_decoder_seed_{seed}.npz"

        save_data(
            data=data,
            path=output_data_path,
            seed=int(seed),
            info=info,
        )
    