import numpy as np
import matplotlib.pyplot as plt
from . import utils
from torch.nn import BCELoss
import deepxde as dde
import torch

"""
In this script, we will train a DeepONet that takes PCA transformed coordinates
as input and outputs the cooresponding pattern. We will only use first 25 co-
ordinates. A subset with efficiency >= 0.9, wavelength = 850nm, and angle = 65 degrees.

DeepONet takes function as input and output another function. The input function is
1D-to-1D, same for the output function.
"""
# Set random seed
seed = 658
dde.config.set_random_seed(seed)

# Load dataset
d = np.load('data/90_all_all_train.npz')
X_train_branch = d['X1'][:,:25].astype(np.float32)
X_train_trunk = d['X2'].astype(np.float32)
y_train = d['y'].astype(np.float32)
d = np.load('data/90_all_all_val.npz')
X_test_branch = d['X1'][:,:25].astype(np.float32)
X_test_trunk = d['X2'].astype(np.float32)
y_test = d['y'].astype(np.float32)


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
loss_fn = CustomLossPlus(alpha = 2)
model.compile("adam", lr=0.001, loss = loss_fn, metrics=[binary_cross_entropy_np])
for i in range(10):
    iterations = 40000
    losshistory, train_state = model.train(iterations=iterations, model_save_path=f"model/model_{seed}.ckpt")
    model.net.scale_factor *= 2


# Plot the loss history
dde.utils.plot_loss_history(losshistory)
plt.legend(['train loss', 'val loss', 'val metric'])
plt.show()

# Save loss history
np.savez(f"model/model_{seed}_losshistory.npz", train_loss=losshistory.loss_train, val_loss=losshistory.loss_test, val_metric=losshistory.metrics_test)
