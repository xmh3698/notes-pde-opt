import argparse
import numpy as np

import subsample_data
from pca_decoder_opt import run_cma_pca_experiment


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run CMA-ES optimization with PCA decoder on subsampled data."
    )

    parser.add_argument("--angle", type=int, required=True)
    parser.add_argument("--wavelength", type=int, required=True)
    parser.add_argument("--seednum", type=int, default=None)
    parser.add_argument("--dataid", type=int, required=True)

    parser.add_argument("--latent_dim", type=int, default=25)
    parser.add_argument("--pop", type=int, default=20)
    parser.add_argument("--total_iters", type=int, default=150)
    parser.add_argument("--sigma0", type=float, default=2.0)
    parser.add_argument("--num_restarts", type=int, default=5)

    return parser.parse_args()


def main():
    args = parse_args()

    angle = args.angle
    wavelength = args.wavelength

    if args.seednum is None:
        seednum = np.random.randint(0, 1000)
        print(f"No seednum provided. Randomly generated seednum: {seednum}")
    else:
        seednum = args.seednum
        print(f"Using provided seednum: {seednum}")

    data_path = f"../../data/subsample/subsample_dataset_seed_{args.dataid}.npz"
    data = np.load(data_path)

    latent, pca = subsample_data.pca_transform_data(
        patterns=data["pattern"],
        efficiencies=data["efficiency"],
        latent_dim=args.latent_dim,
        save_pca_path=f"../../model/subsample_pca/subsample_pca_decoder_seed_{args.dataid}.npz",
    )

    components = pca.components_
    mean = pca.mean_

    run_cma_pca_experiment(
        angle=angle,
        wavelength=wavelength,
        seednum=seednum,
        components=components,
        mean=mean,
        latent_dim=args.latent_dim,
        data_id=args.dataid,
        pop=args.pop,
        total_iters=args.total_iters,
        sigma0=args.sigma0,
        num_restarts=args.num_restarts,
    )


if __name__ == "__main__":
    main()