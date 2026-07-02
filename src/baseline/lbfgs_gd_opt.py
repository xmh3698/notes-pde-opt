# Run L-BFGS optimization with DeepONet surrogate model for a given angle and wavelength
# Then switch to gradient descent optimization with very small learning rate


import json
import time
import math
import numpy as np
import torch
import cma
import matplotlib.pyplot as plt
import deepxde as dde
from src import utils
import argparse
import os

# ---------------------------
# Parse command line arguments
# ---------------------------
parser = argparse.ArgumentParser(description="Run DeepONet + L-BFGS + gradient descent optimization with specified angle and wavelength.")
parser.add_argument("--angle", type=int, required=True, help="Incident angle in degrees")
parser.add_argument("--wavelength", type=int, required=True, help="Wavelength in nm")
parser.add_argument("--seednum", type=int, default=None, help="Seed number for initial starting point")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for gradient descent")
args = parser.parse_args()

angle = args.angle
wavelength = args.wavelength

# If seednum not provided, generate one randomly
if args.seednum is None:
    seednum = np.random.randint(0, 10**3)  # e.g., a random integer up to 1000
    print(f"No seednum provided. Randomly generated seed: {seednum}")
else:
    seednum = args.seednum
    print(f"Using provided seednum: {seednum}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seed
seed = 658 # The seed number for pre-trained neural operator
dde.config.set_random_seed(seed)
utils.set_seed(seed)

# Load dataset
d = np.load('data/90_all_all_train.npz')
X_train_branch = d['X1'][:,:25].astype(np.float32)
X_train_trunk = d['X2'].astype(np.float32)
y_train = d['y'].astype(np.float32)
d = np.load('data/90_all_all_val.npz')
X_test_branch = d['X1'][:,:25].astype(np.float32)
X_test_trunk = d['X2'].astype(np.float32)
y_test = d['y'].astype(np.float32)
X_test_trunk_torch = torch.tensor(X_test_trunk, dtype=torch.float32).to(device).requires_grad_(False)

# Load trained model
# Define the data
data = dde.data.TripleCartesianProd(
    X_train=(X_train_branch, X_train_trunk),
    y_train=y_train,
    X_test=(X_test_branch, X_test_trunk),
    y_test=y_test
)

# Choose a network
m = 25 # number of PCA components
dim_x = 1 # dim of trunk net
net = utils.DeepONetCartesianProdPlus(
    [m, 60, 60, 60],
    [dim_x, 60, 60, 60],
    "relu",
    "Glorot normal",

)
model = dde.Model(data, net)
model.compile(optimizer='adam', lr = 0.001)
model.restore(save_path=f"model/model{seed}/model{seed}.ckpt-400000.pt", verbose=1)
for param in model.net.parameters():
    param.requires_grad = False
model.net.eval()

# Define the loss function
n_Si = 3.70812038
period = abs(wavelength / np.sin(np.radians(angle)))
thickness = 325
#X_test_trunk = torch.tensor(X_test_trunk, dtype=torch.float32).to(device).requires_grad_(False)

def loss_np(x):
    """
    Input x: a 1D numpy array
    Output: a list of scalar
    """
    # if x is 1D, reshape it to 2D
    if x.ndim == 1:
        x = x.reshape(1, -1)
    else:
        raise ValueError("Input x must be a 1D numpy array")
    #x *= 1000 # scale back to original range
    y_pred = model.predict((x, X_test_trunk))
    # Calculate efficiency one by one
    eff = utils.meent_em(angle, n_Si, y_pred, period, thickness, wavelength, backend=0)
    return float(-eff)  

def loss_torch(x):
    """
    Input x: a 1D torch tensor
    Ouput: a gradient vector
    """

    # if x is 1D, reshape it to 2D
    if x.ndim == 1:
        x = x.view(1, -1)
    else:
        raise ValueError("Input x must be a 1D torch tensor")
    y_pred = model.net((x, X_test_trunk_torch))
    eff = utils.meent_em(angle, n_Si, y_pred, period, thickness, wavelength, backend=2)
    return (-eff)

def save_grad_hist(history, path):
    """
    Save the loss curve of the optimization process
    :param history: a list of loss values
    :param path: File path to save the image (e.g., 'output/loss_curve.png').
    """
    dir_path = os.path.dirname(path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)

    plt.figure(figsize=(8, 4), facecolor='white')
    plt.plot(history, color='blue', linewidth=2)
    plt.xlabel('Iteration', fontsize=10)
    plt.ylabel('gradient norm', fontsize=10)
    plt.title('Optimization Curve', fontsize=12)
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close()

print(model.net.scale_factor)


# Start optimization
start_time = time.time()
seed = int(angle * 1e6 + wavelength * 1e2 + seednum)
np.random.seed(seed)
x_sample = np.random.uniform(-1,1,(100, 25)) * 60 # initial guess
total_iters = 200
x_hist = []
results = {'patterns': [], 'efficiencies': [], 'grad_norms': [], 'latents':[]}
results_lbfgs = {'patterns': [], 'efficiencies': [], 'grad_norms': [], 'latents':[]}
opt_hist = []
for i in range(x_sample.shape[0]):
    x0 = torch.tensor(x_sample[i], dtype=torch.float32, requires_grad=True, device=device)
    optimizer = torch.optim.LBFGS([x0],
                                line_search_fn="strong_wolfe",
                                max_iter=total_iters,
                                tolerance_grad=1e-9,
                                tolerance_change=1e-11,)
    grad_norms = []
    print(f"Starting optimization for sample {i}.")
    def closure():
        optimizer.zero_grad()
        loss = loss_torch(x0)
        loss.backward()
        grad_norms.append(x0.grad.norm().item())
        return loss
    loss = optimizer.step(closure)

    x_best = x0.detach().cpu().numpy().reshape(1, -1)
    y_best = model.predict((x_best, X_test_trunk))
    eff_best = utils.meent_em(angle, n_Si, y_best, period, thickness, wavelength, backend=0)
    results_lbfgs['patterns'].append(y_best)
    results_lbfgs['latents'].append(x_best)
    results_lbfgs['efficiencies'].append(eff_best)
    results_lbfgs['grad_norms'].append(grad_norms[-1])

    # After L-BFGS optimization, switch to full-gradient descent
    gd_lr = args.lr
    gd_optimizer = torch.optim.SGD([x0], lr=gd_lr)

    gd_iters = 2000
    for k in range(gd_iters):
        gd_optimizer.zero_grad()
        loss = loss_torch(x0)
        loss.backward()

        grad_norm = x0.grad.norm().item()
        grad_norms.append(grad_norm)

        gd_optimizer.step()


    x_best = x0.detach().cpu().numpy().reshape(1, -1)
    y_best = model.predict((x_best, X_test_trunk))
    eff_best = utils.meent_em(angle, n_Si, y_best, period, thickness, wavelength, backend=0)
    path = f'image/LBFGS_GD/seed{seed}/result{i}.png'
    utils.save_pattern_sequence(y_best, path, eff_best)
    save_grad_hist(grad_norms, f'image/LBFGS_GD/seed{seed}/grad_hist_{i}.png')
    results['patterns'].append(y_best)
    results['latents'].append(x_best)
    results['efficiencies'].append(eff_best)
    results['grad_norms'].append(grad_norms[-1])


end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds for NOTES {seed}")
#breakpoint()


# Save results
np.savez(f'result/lbfgs/LBFGS_{seed}.npz',**results_lbfgs)
np.savez(f'result/lbfgs_gd/LBFGS_GD_{seed}.npz',**results)
np.savez(f'result/opt_hist/LBFGS_GD_opt_hist_{seed}.npz', opt_hist=opt_hist)
#breakpoint()


eff_hist = results['efficiencies']
# Calculate statistics
mean_val = np.mean(eff_hist)
std_val = np.std(eff_hist)
max_val = np.max(eff_hist)

# Plot histogram
plt.figure(figsize=(8, 5))
plt.hist(eff_hist, bins=25, color='skyblue', edgecolor='black')

# Annotate with lines and labels
plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2f}%')
plt.axvline(std_val + mean_val, color='orange', linestyle='dotted', linewidth=2, label=f'Mean + 1 STD: {(mean_val + std_val):.2f}%')
plt.axvline(mean_val - std_val, color='orange', linestyle='dotted', linewidth=2, label=f'Mean - 1 STD: {(mean_val - std_val):.2f}%')
plt.axvline(max_val, color='green', linestyle='solid', linewidth=2, label=f'Max: {max_val:.2f}%')

# Labels and legend
plt.title('Histogram of efficiency')
plt.xlabel('efficiency')
plt.ylabel('Frequency')
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.savefig(f'image/LBFGS_GD/seed{seed}/Figure_1.png')
plt.close()

# Save experimental details
meta = {
    "angle": angle,
    "wavelength": wavelength,
    "seednum": seednum,
    "run_seed": seed,
    "total_iters": total_iters,
    "latent_dim": 25,
    "n_Si": n_Si,
    "thickness": thickness,
    "period": period,
    "grad_descent_lr": gd_lr,
}
with open(f'image/LBFGS_GD/seed{seed}/meta.json', 'w') as f:
    json.dump(meta, f, indent=4)



