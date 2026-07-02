import argparse

from src.notes.data import build_notes_dataset
from src.notes.modeling import NotesModelConfig, build_notes_model, set_all_seeds, train_notes_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train NOTES DeepONet decoder on a subsampled dataset.")
    parser.add_argument("--dataid", type=int, required=True)
    parser.add_argument("--seed", type=int, default=658)
    parser.add_argument("--latent_dim", type=int, default=25)
    parser.add_argument("--data_root", type=str, default="data/subsample")
    parser.add_argument("--output_dir", type=str, default="model/subsample")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=20.0)
    parser.add_argument("--iterations_per_stage", type=int, default=40000)
    parser.add_argument("--num_stages", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    set_all_seeds(args.seed)

    dataset, pca, data_path = build_notes_dataset(
        dataid=args.dataid,
        latent_dim=args.latent_dim,
        data_root=args.data_root,
    )
    print(f"Loaded dataset from {data_path}")
    print(f"Train branch shape: {dataset.X_train_branch.shape}")
    print(f"Train output shape: {dataset.y_train.shape}")

    model_config = NotesModelConfig(latent_dim=args.latent_dim)
    model = build_notes_model(dataset, model_config)

    train_notes_model(
        model=model,
        seed=args.seed,
        output_dir=args.output_dir,
        lr=args.lr,
        alpha=args.alpha,
        iterations_per_stage=args.iterations_per_stage,
        num_stages=args.num_stages,
    )


if __name__ == "__main__":
    main()
