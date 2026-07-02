import argparse
import os

import numpy as np
import torch

from src.notes.data import build_notes_dataset
from src.notes.modeling import NotesModelConfig, load_trained_notes_model, set_all_seeds
from src.notes.optimization import NotesOptimizationConfig, run_notes_cmaes


def parse_args():
    parser = argparse.ArgumentParser(description="Run NOTES + CMA-ES optimization.")
    parser.add_argument("--angle", type=int, required=True)
    parser.add_argument("--wavelength", type=int, required=True)
    parser.add_argument("--dataid", type=int, required=True)
    parser.add_argument("--seednum", type=int, default=None)
    parser.add_argument("--model_seed", type=int, default=658)
    parser.add_argument("--latent_dim", type=int, default=25)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data_root", type=str, default="data/subsample")
    parser.add_argument("--result_dir", type=str, default="result/notes")
    parser.add_argument("--image_root", type=str, default="image/notes/CMA")
    parser.add_argument("--popsize", type=int, default=20)
    parser.add_argument("--total_iters", type=int, default=150)
    parser.add_argument("--num_restarts", type=int, default=5)
    parser.add_argument("--sigma0", type=float, default=1000.0)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.seednum is None:
        seednum = int(np.random.randint(0, 1000))
        print(f"No seednum provided. Randomly generated seednum: {seednum}")
    else:
        seednum = args.seednum
        print(f"Using provided seednum: {seednum}")

    set_all_seeds(args.model_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset, _, data_path = build_notes_dataset(
        dataid=args.dataid,
        latent_dim=args.latent_dim,
        data_root=args.data_root,
    )
    print(f"Loaded dataset from {data_path}")

    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = f"model/subsample/model{args.model_seed}/model{args.model_seed}.ckpt-400000.pt"

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Cannot find checkpoint: {checkpoint}")

    model_config = NotesModelConfig(latent_dim=args.latent_dim)
    model = load_trained_notes_model(
        dataset=dataset,
        config=model_config,
        checkpoint_path=checkpoint,
        device=device,
    )

    opt_config = NotesOptimizationConfig(
        latent_dim=args.latent_dim,
        sigma0=args.sigma0,
        popsize=args.popsize,
        total_iters=args.total_iters,
        num_restarts=args.num_restarts,
    )

    run_notes_cmaes(
        model=model,
        trunk_input=dataset.X_test_trunk,
        angle=args.angle,
        wavelength=args.wavelength,
        seednum=seednum,
        dataid=args.dataid,
        config=opt_config,
        result_dir=args.result_dir,
        image_root=args.image_root,
    )


if __name__ == "__main__":
    main()
