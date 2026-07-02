# Compute gradient norm of existing optimization results
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
parser = argparse.ArgumentParser(description="Compute gradient norm of existing optimization results.")
parser.add_argument("--angle", type=int, required=True, help="Incident angle in degrees")
parser.add_argument("--wavelength", type=int, required=True, help="Wavelength in nm")
parser.add_argument("--data_path", type=str, required=True, help="Path to the data file")
args = parser.parse_args()

angle = args.angle
wavelength = args.wavelength

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
model.net.to(device)
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

    y_pred = model.net((x, X_test_trunk_torch))
    eff = utils.meent_em(angle, n_Si, y_pred, period, thickness, wavelength, backend=2)
    return (-eff)

print(model.net.scale_factor)


# Start optimization
data_path = args.data_path
data = np.load(args.data_path)
x_sample = data['latents'].astype(np.float32)
start_time = time.time()
results = {'grad_norms': []}
opt_hist = []
for i in range(x_sample.shape[0]):
    x0 = torch.tensor(x_sample[i], dtype=torch.float32, requires_grad=True, device=device)
    loss = loss_torch(x0)

    if x0.grad is not None:
        x0.grad.zero_()

    loss.backward()
    grad_norm = x0.grad.norm().item()
    results["grad_norms"].append(grad_norm)
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds for NOTES {seed}")
#breakpoint()


# Save results
save_path = args.data_path.replace(".npz", "_wg.npz")
np.savez(save_path, **data,**results)
print(f"Saved to {save_path}")
