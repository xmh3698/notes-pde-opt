import os
import time
import json
import numpy as np
import cma
import meent
import matplotlib.pyplot as plt


def sigmoid_np(x, beta=10, eta=0):
    return 1.0 / (1.0 + np.exp(-beta * (x - eta)))


def save_pattern_sequence(sequence, path, efficiency):
    sequence = np.asarray(sequence)

    if sequence.ndim == 2 and sequence.shape[0] == 1:
        sequence = sequence[0]
    elif sequence.ndim != 1:
        raise ValueError("Input sequence must be 1D or 2D with one row.")

    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    filename = os.path.splitext(os.path.basename(path))[0]
    path = os.path.join(dir_path, f"{filename}_eff{efficiency:.4f}.png")

    plt.figure(figsize=(8, 4), facecolor="white")
    plt.bar(range(len(sequence)), sequence, color="black", edgecolor="black")
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_frame_on(False)

    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.title(f"Efficiency: {efficiency:.4f}", fontsize=12)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close()


def meent_em(angle, n_Si, pattern, period, thickness, wavelength, backend=0):
    pattern = np.asarray(pattern)

    if pattern.ndim == 1:
        pattern_dim = pattern.shape[0]
    elif pattern.ndim == 2 and pattern.shape[0] == 1:
        pattern_dim = pattern.shape[1]
    else:
        raise ValueError("pattern must be shape (d,) or (1, d).")

    angle_rad = np.radians(angle)

    if np.sin(angle_rad) != 0:
        expected_period = abs(wavelength / np.sin(angle_rad))
        assert np.isclose(period, expected_period, rtol=1e-5), (
            f"Period {period} does not match expected period {expected_period}"
        )

    pol = 1
    n_top = 1.45
    n_bot = 1.0
    theta = 0.0
    fto = [40]

    thickness = [thickness]
    period = [period]

    ucell_1d_s = pattern.reshape((1, 1, pattern_dim)) * (n_Si - 1) + 1

    mee = meent.call_mee(
        backend=backend,
        pol=pol,
        n_top=n_top,
        n_bot=n_bot,
        theta=theta,
        fto=fto,
        wavelength=wavelength,
        period=period,
        ucell=ucell_1d_s,
        thickness=thickness,
        type_complex=np.complex128,
    )

    result = mee.conv_solve()
    de_ti = result.res.de_ti

    return float(de_ti[0, fto[0] + 1])


def make_loss_fn(
    angle,
    wavelength,
    components,
    mean,
    latent_dim=25,
    beta=10,
    eta=0,
    reg_weight=0.01,
):
    n_Si = 3.70812038
    period = abs(wavelength / np.sin(np.radians(angle)))
    thickness = 325

    def loss_np(x):
        x = np.asarray(x)

        if x.ndim == 1:
            x = x.reshape(1, -1)
        elif x.ndim != 2 or x.shape[0] != 1:
            raise ValueError(f"Expected shape (d,) or (1, d), got {x.shape}")

        y_pred = x @ components[:latent_dim, :] + mean
        y_pred = sigmoid_np(y_pred, beta=beta, eta=eta)

        eff = meent_em(
            angle=angle,
            n_Si=n_Si,
            pattern=y_pred,
            period=period,
            thickness=thickness,
            wavelength=wavelength,
            backend=0,
        )

        reg = np.sum(y_pred * (1.0 - y_pred))

        return float(-eff + reg_weight * reg)

    return loss_np


def run_cma_pca_experiment(
    angle,
    wavelength,
    seednum,
    components,
    mean,
    latent_dim=25,
    data_id=None,
    output_root="../../image/subsample/CMA_PCA",
    result_dir="../../result/subsample",
    pop=20,
    total_iters=150,
    sigma0=2.0,
    num_restarts=5,
):
    start_time = time.time()

    seed = int(angle * 1e6 + wavelength * 1e2 + seednum)
    rng = np.random.default_rng(seed)

    n_Si = 3.70812038
    period = abs(wavelength / np.sin(np.radians(angle)))
    thickness = 325

    os.makedirs(result_dir, exist_ok=True)
    image_dir = os.path.join(output_root, f"seed{seed}")
    os.makedirs(image_dir, exist_ok=True)

    loss_np = make_loss_fn(
        angle=angle,
        wavelength=wavelength,
        components=components,
        mean=mean,
        latent_dim=latent_dim,
    )

    x_sample = rng.standard_normal((num_restarts, latent_dim))

    results = {
        "patterns": [],
        "efficiencies": [],
    }

    opt_hist = []

    for i in range(num_restarts):
        x0 = x_sample[i]
        initial_loss = loss_np(x0)

        x_best, es = cma.fmin2(
            loss_np,
            x0,
            sigma0,
            {
                "popsize": pop,
                "tolfun": 1e-6,
                "maxiter": total_iters,
                "tolflatfitness": 3,
            },
        )

        x_best = x_best.reshape(1, -1)
        y_best = x_best @ components[:latent_dim, :] + mean
        y_best = sigmoid_np(y_best, beta=10, eta=0)

        eff_best = meent_em(
            angle=angle,
            n_Si=n_Si,
            pattern=y_best,
            period=period,
            thickness=thickness,
            wavelength=wavelength,
            backend=0,
        )

        save_pattern_sequence(
            y_best,
            os.path.join(image_dir, f"pop{pop}_{i}.png"),
            eff_best,
        )

        results["patterns"].append(y_best)
        results["efficiencies"].append(eff_best)

        log_data = es.logger.load().data
        fvals = log_data["f"][:, 4]
        fvals = np.insert(fvals, 0, initial_loss)

        if len(fvals) < total_iters + 1:
            fvals = np.pad(fvals, (0, total_iters + 1 - len(fvals)), mode="edge")

        num_evals = np.arange(total_iters + 1) * pop
        opt_hist.append((num_evals, fvals))

    results["patterns"] = np.asarray(results["patterns"])
    results["efficiencies"] = np.asarray(results["efficiencies"])

    result_path = os.path.join(result_dir, f"CMA_PCA_{seed}.npz")
    hist_path = os.path.join(result_dir, f"CMA_PCA_opt_hist_{seed}.npz")

    np.savez(result_path, **results)
    np.savez(hist_path, opt_hist=np.asarray(opt_hist, dtype=object))

    plot_efficiency_histogram(
        results["efficiencies"],
        os.path.join(image_dir, "efficiency_histogram.png"),
    )

    meta = {
        "angle": angle,
        "wavelength": wavelength,
        "seednum": seednum,
        "run_seed": seed,
        "data_id": data_id,
        "popsize": pop,
        "sigma0": sigma0,
        "total_iters": total_iters,
        "latent_dim": latent_dim,
        "n_Si": n_Si,
        "thickness": thickness,
        "period": period,
        "num_restarts": num_restarts,
        "result_path": result_path,
        "hist_path": hist_path,
    }

    with open(os.path.join(image_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=4)

    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds for PCA+CMA-ES {seed}")

    return results, opt_hist, meta


def plot_efficiency_histogram(efficiencies, path):
    efficiencies = np.asarray(efficiencies)

    mean_val = np.mean(efficiencies)
    std_val = np.std(efficiencies)
    max_val = np.max(efficiencies)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.hist(efficiencies, bins=25)

    plt.axvline(mean_val, linestyle="dashed", linewidth=2, label=f"Mean: {mean_val:.4f}")
    plt.axvline(mean_val + std_val, linestyle="dotted", linewidth=2, label=f"Mean + STD: {mean_val + std_val:.4f}")
    plt.axvline(mean_val - std_val, linestyle="dotted", linewidth=2, label=f"Mean - STD: {mean_val - std_val:.4f}")
    plt.axvline(max_val, linestyle="solid", linewidth=2, label=f"Max: {max_val:.4f}")

    plt.title("Histogram of Efficiency")
    plt.xlabel("Efficiency")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()