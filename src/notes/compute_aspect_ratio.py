from numpy.random import Generator, PCG64
import numpy as np
import matplotlib.pyplot as plt
from src import utils
import re
import argparse
import os

def smallest_gap_aspect_ratio(pattern, period, thickness=325):
    pattern = np.asarray(pattern).squeeze()

    if pattern.ndim != 1:
        raise ValueError(f"Expected 1D pattern, got shape {pattern.shape}")

    pattern = (pattern > 0.5).astype(int)
    pixel_size = period / len(pattern)

    diff = np.diff(np.concatenate(([1], pattern, [1])))

    zero_starts = np.where(diff == -1)[0]
    zero_ends = np.where(diff == 1)[0]
    zero_lengths = zero_ends - zero_starts

    if len(zero_lengths) == 0:
        return np.nan

    min_gap_pixels = np.min(zero_lengths)
    min_gap_physical = min_gap_pixels * pixel_size

    return thickness / min_gap_physical

def average_aspect_ratios(method_name, seednum, wavelength, angle):
    period = abs(wavelength / np.sin(np.radians(angle)))
    path = f'result/{method_name}_{seednum}.npz'
    d = np.load(path)
    patterns = d['patterns']
    aspect_ratios = [
        smallest_gap_aspect_ratio(p, period)
        for p in patterns
    ]
    # Compute average
    # Return the average aspect ratio and optionally its std deviation
    avg_ratio = np.mean(aspect_ratios)
    std_ratio = np.std(aspect_ratios)

    print(f"The average aspect ratio is {avg_ratio:.4f} (± {std_ratio:.4f})")
    return avg_ratio, std_ratio

def compare_aspect_ratios(wavelength, angle, seednum, thickness=325):
    """
    Compute and compare the average aspect ratio between CMA-ES and baseline patterns.
    """
    key = int(angle * 1e6 + wavelength * 1e2 + seednum)

    period = abs(wavelength / np.sin(np.radians(angle)))

    # Load CMA-ES results
    path_cmaes = f'result/CMA-ES_{key}.npz'
    d = np.load(path_cmaes)
    patterns_cmaes = d['patterns']

    # Load baseline results
    path_base = f'result/baseline_{key}.npz'
    d = np.load(path_base)
    patterns_baseline = d['patterns']

    # Compute aspect ratios for each pattern
    aspect_ratios_cmaes = [
        smallest_gap_aspect_ratio(p, period)
        for p in patterns_cmaes
    ]
    aspect_ratios_baseline = [
        smallest_gap_aspect_ratio(p, period)
        for p in patterns_baseline
    ]

    # Compute averages
    avg_cmaes = np.mean(aspect_ratios_cmaes)
    avg_baseline = np.mean(aspect_ratios_baseline)
    difference = avg_cmaes - avg_baseline

    # Print comparison summary
    print(f"Wavelength: {wavelength:.1f} nm | Angle: {angle:.1f}° | Seed: {seednum}")
    print(f"  NOTES   avg aspect ratio: {avg_cmaes:.3f}")
    print(f"  Baseline avg aspect ratio: {avg_baseline:.3f}")
    print(f"  Difference (NOTES - Baseline): {avg_cmaes - avg_baseline:.3f}")

    return aspect_ratios_cmaes, aspect_ratios_baseline, difference

def compute_aspect_ratios(wavelength, angle, seednum, thickness=325):
    """
    Compute and aspect ratio
    """
    key = int(angle * 1e6 + wavelength * 1e2 + seednum)

    period = abs(wavelength / np.sin(np.radians(angle)))

    # Load CMA-ES results
    path_cmaes = f'result/CMA-ES_{key}.npz'
    d = np.load(path_cmaes)
    patterns_cmaes = d['patterns']

    # Compute aspect ratios for each pattern
    aspect_ratios_cmaes = [
        smallest_gap_aspect_ratio(p, period)
        for p in patterns_cmaes
    ]
    return aspect_ratios_cmaes

def compute_aspect_ratio_from_best_designs(path):
    """
    Compute aspect ratios from the best designs in the given path.
    """
    d = np.load(path)
    effs = d['efficiencies']
    patterns = d['patterns']

    # Extract the number from filename
    number = re.search(r'\d+', path).group()

    angle = int(number[0:2])
    wavelength = int(number[2:5]) * 10
    print(f"Angle: {angle}, Wavelength: {wavelength}")
    period = abs(wavelength / np.sin(np.radians(angle)))

    eff_hist = effs
    max_index = np.argmax(eff_hist)
    utils.visualize_pattern_sequence(patterns[max_index], eff_hist[max_index])
    aspect_ratio = smallest_gap_aspect_ratio(patterns[max_index], period)
    print(f"Aspect ratio of best design: {aspect_ratio:.5f}")

def compare_aspect_ratios_histogram(cma_as, base_as):
    """
    Compare aspect ratio distributions between NOTES (cma_as) and Baseline (base_as).
    
    Parameters
    ----------
    cma_as : array-like
        1D or 2D array of aspect ratios from NOTES.
    base_as : array-like
        1D or 2D array of aspect ratios from Baseline.
    """
    # Flatten in case of 2D arrays
    cma_as = np.asarray(cma_as).flatten()
    base_as = np.asarray(base_as).flatten()

    # Remove NaN or inf if present
    cma_as = cma_as[np.isfinite(cma_as)]
    base_as = base_as[np.isfinite(base_as)]

    # Plot histogram comparison
    plt.figure(figsize=(7, 5))
    bins = np.linspace(
        min(cma_as.min(), base_as.min()),
        max(cma_as.max(), base_as.max()),
        25
    )
    plt.hist(cma_as, bins=bins, alpha=1, label='NOTES (CMA-ES)', color='tab:blue', density=True)
    plt.hist(base_as, bins=bins, alpha=0.6, label='Baseline (Direct CMA-ES)', color='tab:orange', density=True)

    plt.xlabel('Aspect Ratio', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Comparison of Aspect Ratio Distributions', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()



# ---------------------------
# Parse command line arguments
# ---------------------------
parser = argparse.ArgumentParser(description="Run DeepONet + L-BFGS optimization with specified angle and wavelength.")
parser.add_argument("--angle", type=int, required=True, help="Incident angle in degrees")
parser.add_argument("--wavelength", type=int, required=True, help="Wavelength in nm")
parser.add_argument("--data_path", type=str, required=True, help="Path to the data file")
parser.add_argument("--save_path", type=str, required=True, help="Path to save the results")
args = parser.parse_args()

angle = args.angle
wavelength = args.wavelength
data_path = args.data_path
save_path = args.save_path

designs = np.load(data_path)['patterns']
period = abs(wavelength / np.sin(np.radians(angle)))

aspect_ratios = [
    smallest_gap_aspect_ratio(design, period)
    for design in designs
]

sorted_ratios = np.sort(aspect_ratios)

# Compute cumulative probabilities
cdf = np.arange(1, len(sorted_ratios) + 1) / len(sorted_ratios)

# Plot
plt.figure()
plt.plot(sorted_ratios, cdf)
plt.xlabel("Aspect Ratio")
plt.ylabel("Cumulative Probability")
plt.title("Empirical CDF of Aspect Ratios")
plt.vlines(33, 0.05, 0.95, colors='red', linestyles='dashed')
plt.grid(True)
plt.savefig(f"{save_path}/aspect_ratio_cdf.png", bbox_inches='tight')