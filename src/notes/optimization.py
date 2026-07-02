import os
import time
from dataclasses import dataclass

import numpy as np
import cma
import matplotlib.pyplot as plt

from src import utils


@dataclass
class NotesOptimizationConfig:
    latent_dim: int = 25
    n_Si: float = 3.70812038
    thickness: float = 325.0
    sigma0: float = 1000.0
    popsize: int = 20
    total_iters: int = 150
    num_restarts: int = 30
    init_low: float = -60.0
    init_high: float = 60.0
    tolfun: float = 1e-6
    tolflatfitness: int = 3


def compute_period(angle: float, wavelength: float):
    return abs(wavelength / np.sin(np.radians(angle)))


def make_notes_loss(model, trunk_input, angle, wavelength, config: NotesOptimizationConfig):
    period = compute_period(angle, wavelength)

    def loss_np(x):
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        elif x.ndim != 2 or x.shape[0] != 1:
            raise ValueError(f"Expected shape (d,) or (1, d), got {x.shape}")

        y_pred = model.predict((x, trunk_input))
        eff = utils.meent_em(
            angle,
            config.n_Si,
            y_pred,
            period,
            config.thickness,
            wavelength,
            backend=0,
        )
        return float(-eff)

    return loss_np


def run_notes_cmaes(
    model,
    trunk_input,
    angle: int,
    wavelength: int,
    seednum: int,
    dataid: int,
    config: NotesOptimizationConfig,
    result_dir: str = "result/notes",
    image_root: str = "image/notes/CMA",
):
    start_time = time.time()
    run_seed = int(angle * 1e6 + wavelength * 1e2 + seednum)
    rng = np.random.default_rng(run_seed)

    os.makedirs(result_dir, exist_ok=True)
    hist_dir = os.path.join(result_dir, "opt_hist")
    os.makedirs(hist_dir, exist_ok=True)

    image_dir = os.path.join(image_root, f"seed{run_seed}")
    os.makedirs(image_dir, exist_ok=True)

    loss_np = make_notes_loss(model, trunk_input, angle, wavelength, config)
    x_sample = rng.uniform(
        config.init_low,
        config.init_high,
        size=(config.num_restarts, config.latent_dim),
    )

    period = compute_period(angle, wavelength)
    results = {"patterns": [], "efficiencies": [], "latents": []}
    opt_hist = []

    for i, x0 in enumerate(x_sample):
        print(f"CMA-ES restart {i + 1}/{config.num_restarts}")
        initial_loss = loss_np(x0)

        x_best, es = cma.fmin2(
            loss_np,
            x0,
            config.sigma0,
            {
                "popsize": config.popsize,
                "tolfun": config.tolfun,
                "maxiter": config.total_iters,
                "tolflatfitness": config.tolflatfitness,
            },
        )

        x_best = x_best.reshape(1, -1)
        y_best = model.predict((x_best, trunk_input))
        eff_best = utils.meent_em(
            angle,
            config.n_Si,
            y_best,
            period,
            config.thickness,
            wavelength,
            backend=0,
        )

        utils.save_pattern_sequence(
            y_best,
            os.path.join(image_dir, f"pop{config.popsize}_{i}.png"),
            eff_best,
        )

        results["patterns"].append(y_best)
        results["latents"].append(x_best)
        results["efficiencies"].append(eff_best)

        log_data = es.logger.load().data
        fvals = log_data["f"][:, 4]
        fvals = np.insert(fvals, 0, initial_loss)
        if len(fvals) < config.total_iters + 1:
            fvals = np.pad(fvals, (0, config.total_iters + 1 - len(fvals)), mode="edge")

        num_evals = np.arange(config.total_iters + 1) * config.popsize
        opt_hist.append((num_evals, fvals))

    results = {k: np.asarray(v) for k, v in results.items()}

    result_path = os.path.join(result_dir, f"CMA-ES_{run_seed}.npz")
    hist_path = os.path.join(hist_dir, f"CMA-ES_opt_hist_{run_seed}.npz")
    np.savez(result_path, **results)
    np.savez(hist_path, opt_hist=np.asarray(opt_hist, dtype=object))

    plot_efficiency_histogram(
        results["efficiencies"],
        os.path.join(image_dir, "efficiency_histogram.png"),
    )

    print(f"Saved results to {result_path}")
    print(f"Saved optimization history to {hist_path}")
    print(f"Time taken: {time.time() - start_time:.2f} seconds for NOTES {run_seed}")

    return results, opt_hist


def plot_efficiency_histogram(efficiencies, path):
    efficiencies = np.asarray(efficiencies)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    mean_val = np.mean(efficiencies)
    std_val = np.std(efficiencies)
    max_val = np.max(efficiencies)

    plt.figure(figsize=(8, 5))
    plt.hist(efficiencies, bins=25)
    plt.axvline(mean_val, linestyle="dashed", linewidth=2, label=f"Mean: {mean_val:.4f}")
    plt.axvline(mean_val + std_val, linestyle="dotted", linewidth=2, label=f"Mean + STD: {mean_val + std_val:.4f}")
    plt.axvline(mean_val - std_val, linestyle="dotted", linewidth=2, label=f"Mean - STD: {mean_val - std_val:.4f}")
    plt.axvline(max_val, linestyle="solid", linewidth=2, label=f"Max: {max_val:.4f}")
    plt.title("Histogram of efficiency")
    plt.xlabel("Efficiency")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
