import numpy as np
import matplotlib.pyplot as plt
from torch.nn import BCELoss
import deepxde as dde
import torch
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

"""
In this script, we will train a DeepONet that takes PCA transformed coordinates
as input and outputs the cooresponding pattern. We will only use first 25 co-
ordinates. 

DeepONet takes function as input and output another function. The input function is
1D-to-1D, same for the output function.
"""


# Load dataset
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

# Define the data
data = dde.data.TripleCartesianProd(
    X_train=(X_train_branch, X_train_trunk),
    y_train=y_train,
    X_test=(X_test_branch, X_test_trunk),
    y_test=y_test
)

# Choose a network
m = 25 # number of index
dim_x = 1 # dim of trunk net
net = utils.DeepONetCartesianProdPlus(
    [m, 60, 60, 60],
    [dim_x, 60, 60, 60],
    "relu",
    "Glorot normal",
    scale_factor=1,
    bias_factor = 0.5366,
    output_activation = "sigmoid"
)

# Define the loss function
class CustomLoss():
    def __init__(self, alpha = 1, beta = 10):
        self.alpha = alpha
        self.beta = beta
        self.loss_fn = utils.binary_cross_entropy
    def __call__(self, y_true, y_pred):
        y_pred_binary = torch.sigmoid(y_pred*self.beta)  # apply sigmoid to the output
        loss = self.loss_fn(y_true, y_pred_binary)
        reg = self.alpha * y_pred_binary * (1 - y_pred_binary) # add alpha constant
        return loss + reg.mean()
    def eval(self, y_true, y_pred):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)
        y_true = torch.tensor(y_true, dtype=torch.float32)
        y_pred_binary = torch.sigmoid(y_pred*self.beta)
        loss = self.loss_fn(y_true, y_pred_binary)
        return loss.item()

def sigmoid_func(x, beta, eta):
    return torch.sigmoid(beta * (x - eta))

def binary_cross_entropy(y_true, y_pred):
    """
    Compute binary cross-entropy loss.
    :param y_true: True labels.
    :param y_pred: Predicted labels.
    """
    y_pred = torch.clamp(y_pred, 1e-7, 1 - 1e-7)  # Clamping to avoid log(0)
    return -torch.mean(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))

def binary_cross_entropy_np(y_true, y_pred):
    """
    Compute binary cross-entropy loss.
    :param y_true: True labels.
    :param y_pred: Predicted labels.
    """
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)  # Use clip(), not clamp()
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

class CustomLossPlus():
    def __init__(self, alpha = 1):
        self.alpha = alpha
        self.loss_fn = utils.binary_cross_entropy
    def __call__(self, y_true, y_pred):
        loss = self.loss_fn(y_true, y_pred)
        reg = self.alpha * y_pred * (1 - y_pred)
        return loss + reg.mean()
    def eval(self, y_true, y_pred):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)
        y_true = torch.tensor(y_true, dtype=torch.float32)
        loss = self.loss_fn(y_true, y_pred)
        return loss.item()

# Define class beta scheduler
class BetaScheduler(dde.callbacks.Callback):
    def __init__(self):
        super().__init__()
        

    def on_epoch_end(self):
        if self.model.train_state.epoch % 40000 == 0:
            self.model.net.scale_factor *= 2

# Define the model
model = dde.Model(data, net)
# Compile and Train
loss_fn = CustomLossPlus(alpha = 20)
model.compile("adam", lr=0.001, loss = loss_fn, metrics=[binary_cross_entropy_np])
for i in range(10):
    iterations = 40000
    losshistory, train_state = model.train(iterations=iterations, model_save_path=f"model/model_ab{seed}.ckpt")
    model.net.scale_factor *= 2


# Plot the loss history
dde.utils.plot_loss_history(losshistory)
plt.legend(['train loss', 'val loss', 'val metric'])
plt.show()
breakpoint()

# Save loss history
np.savez(f"model/model_ab{seed}_losshistory.npz", train_loss=losshistory.loss_train, val_loss=losshistory.loss_test, val_metric=losshistory.metrics_test)
breakpoint()
