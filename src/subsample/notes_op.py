import time
import math
import numpy as np
import torch
import cma
import matplotlib.pyplot as plt
import deepxde as dde
from src import utils
import subsample_data
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import argparse

# ---------------------------
# Parse command line arguments
# ---------------------------
parser = argparse.ArgumentParser(description="Run CMA-ES optimization with specified angle and wavelength.")
parser.add_argument("--angle", type=int, required=True, help="Incident angle in degrees")
parser.add_argument("--wavelength", type=int, required=True, help="Wavelength in nm")
parser.add_argument("--seednum", type=int, default=None, help="Seed number for initial starting point")
parser.add_argument("--dataid", type=int, required=True)
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
data_path = f"data/subsample/subsample_dataset_seed_{args.dataid}.npz"
d = np.load(data_path)
dataset_pattern = d['pattern']
dataset_eff = d['efficiency']
dataset_pattern = utils.preprocessing(dataset_eff)

latents, _ = subsample_data.pca_transform_data(dataset_pattern,dataset_eff)
y = (dataset_pattern + 1) / 2
X_train_branch, X_test_branch, y_train, y_test = train_test_split(latents, y, train_size=0.8, random_state=42)
X_train_trunk = np.array([[i] for i in range(len(y_train[0]))]) / float(len(y_train[0]))
X_test_trunk = np.array([[i] for i in range(len(y_train[0]))]) / float(len(y_train[0]))
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
model.restore(save_path=f"model/subsample/model{seed}/model{seed}.ckpt-400000.pt", verbose=1)
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

print(model.net.scale_factor)

"""
# Check eff in training set
start = time.time()
effs_train = []
for i in range(y_train.shape[0]):
    y = y_train[i:i+1]
    eff = utils.meent_em(angle, n_Si, y, period, thickness, wavelength, backend=0)
    effs_train.append(eff)
end = time.time()
print(f"Max eff in training set: {max(effs_train)*100:.2f}%")
print(f"Mean eff in training set: {np.mean(effs_train)*100:.2f}%")
print(f"Time taken to compute training set efficiencies: {end - start} seconds")
print(f"Time per sample: {(end - start)/len(effs_train)} seconds")
breakpoint()
"""

# Start optimization
start_time = time.time()
seed = int(angle * 1e6 + wavelength * 1e2 + seednum)
np.random.seed(seed)
x_sample = np.random.uniform(-1,1,(30,25)) * 60 # initial guess
sigma0 = 1000
pop = 20
total_iters = 150
x_hist = []
results = {'patterns': [], 'efficiencies': [], 'latents':[]}
opt_hist = []
for i in range(x_sample.shape[0]):
    x0 = x_sample[i]
    initial_loss = loss_np(x0)
    x, es = cma.fmin2(loss_np, x0, sigma0,
                      {'popsize': pop, 'tolfun': 1e-6, 'maxiter': total_iters, 'tolflatfitness': 3})
    x_best = x.reshape(1, -1)
    y_best = model.predict((x_best, X_test_trunk))
    eff_best = utils.meent_em(angle, n_Si, y_best, period, thickness, wavelength, backend=0)
    path = f'../DeepONet/imag/CMA/seed{seed}/pop{pop}_{i}.png'
    utils.save_pattern_sequence(y_best, path, eff_best)
    results['patterns'].append(y_best)
    results['latents'].append(x_best)
    results['efficiencies'].append(eff_best)

    data = es.logger.load().data
    fvals = data['f'][:, 4]       # best function value
    fvals = np.insert(fvals, 0, initial_loss)
    if len(fvals) < total_iters + 1:
        fvals = np.pad(fvals, (0, total_iters +1 - len(fvals)), 'edge')
    num_favals = np.arange(total_iters+1) * pop  # number of function evaluations
    opt_hist.append((num_favals, fvals))


end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds for NOTES {seed}")
#breakpoint()


# Save results
np.savez(f'result/CMA-ES_{seed}.npz',**results)
np.savez(f'result/opt_hist/CMA-ES_opt_hist_{seed}.npz', opt_hist=opt_hist)
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
plt.savefig(f'../DeepONet/imag/CMA/seed{seed}/Figure_1.png')
plt.close()




