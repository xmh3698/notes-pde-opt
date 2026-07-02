import h5py
import os
import math
import collections
import random
import numpy as np
import numpy.matlib
from numpy import sqrt, pi
import matplotlib.pyplot as plt
import torch
import meent
from deepxde.nn import DeepONetCartesianProd
from deepxde.nn import activations
from scipy.optimize import OptimizeResult
from scipy.special import roots_hermite
from scipy.stats import ortho_group, special_ortho_group

def set_seed(seed):
    """Sets the seed for reproducibility across various libraries.
    :param seed: The seed value to ensure reproducibility
    :return: None
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_file_keys(file_path):
    """
    Retrieve the keys of an .mat file.
    :param file_path: Path to the .mat file.
    """
    keys = []
    with h5py.File(file_path, 'r') as file:
        for key in file.keys():
            #print(key)
            keys.append(key)
    return keys


def load_mat_data(file_path, keyword):
    """
    Load data from a .mat file given a keyword.
    :param file_path: Path to the .mat file.
    :param keyword: Keyword to access the data.
    """
    with h5py.File(file_path, 'r') as file:
        if keyword in file:
            data = file[keyword][()]
            return np.array(data).squeeze()
        else:
            raise KeyError(f"Key '{keyword}' not found in the file.")
        
def npz_filter_data(
        data, 
        efficiency, 
        wavelength_range = [0.0,1400.0], 
        angle_range = [0.0, 90.0]):
    """
    Filter data based on efficiency, wavelength, and angle from npz file.
    :param data: Dataset.
    :param efficiency: Threshold efficiency values.
    :param wavelength_range: Wavelength range.
    :param angle_range: Angle range.
    """
    result = dict()
    mask = ((data['efficiency'] >= efficiency) & (data['wavelength'] <= wavelength_range[1]) & 
            (data['angle'] <= angle_range[1]) & (data['wavelength'] >= wavelength_range[0]) &
            (data['angle'] >= angle_range[0]))
    for key in data.files:
        result[key] = data[key][mask]
    return result
        
def visualize_pattern_sequence(sequence, efficiency):
    """
    Visualize a design pattern in 2D plot
    :param sequence: Design pattern sequence.
    """
    # Convert tensor to NumPy array if needed
    if isinstance(sequence, torch.Tensor):
        sequence = sequence.detach().cpu().numpy()

    # Validate or squeeze shape
    if sequence.ndim == 1:
        pass
    elif sequence.ndim == 2:
        if sequence.shape[0] == 1:
            sequence = sequence[0]
        else:
            raise ValueError("Input sequence must be 1D or 2D with a single row.")
    else:
        raise ValueError("Input sequence must be 1D or 2D.")

    # Plotting
    indices = range(len(sequence))
    plt.figure(figsize=(8, 4), facecolor='white')
    plt.bar(indices, sequence, color='black', edgecolor='black')
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_frame_on(False)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.title(f"Efficiency: {efficiency:.2f}", fontsize=12)
    plt.tight_layout()
    plt.show()

def save_pattern_sequence(sequence, path, efficiency):
    """
    Save a visualization of a design pattern sequence to a file.
    
    :param sequence: 1D or 2D design pattern (NumPy array or PyTorch tensor).
    :param path: File path to save the image.
    :param efficiency: Scalar value to display on the figure.
    """
    # Convert tensor to NumPy array if needed
    if isinstance(sequence, torch.Tensor):
        sequence = sequence.detach().cpu().numpy()

    # Validate or squeeze shape
    if sequence.ndim == 1:
        pass
    elif sequence.ndim == 2:
        if sequence.shape[0] == 1:
            sequence = sequence[0]
        else:
            raise ValueError("Input sequence must be 1D or 2D with a single row.")
    else:
        raise ValueError("Input sequence must be 1D or 2D.")

    # Ensure save directory exists
    dir_path = os.path.dirname(path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)
    filename = os.path.splitext(os.path.basename(path))[0]
    path = os.path.join(dir_path, f"{filename}_eff{efficiency:.2f}.png")
    # Plotting
    indices = range(len(sequence))
    plt.figure(figsize=(8, 4), facecolor='white')
    plt.bar(indices, sequence, color='black', edgecolor='black')
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_frame_on(False)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.title(f"Efficiency: {efficiency:.2f}", fontsize=12)
    plt.tight_layout()

    # Save
    plt.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close()

def save_optimization_hist(history, path):
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
    plt.ylabel('Eff', fontsize=10)
    plt.title('Optimization Curve', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close()


def visualize_fft(sequence):
    """
    Apply FFT to a sequence and visualize the magnitude and angle parts.
    :param sequence: Design pattern sequence.
    """
    fxt = np.fft.fft(sequence)
    plt.subplot(1,2,1)
    plt.plot(np.fft.fftshift(np.abs(fxt)))
    plt.title('Plot of the magnitude of FT of X(t)')
    plt.xlabel('Sample index')
    plt.ylabel('Amplitude')
    plt.subplot(1,2,2)
    plt.plot(np.fft.fftshift(np.angle(fxt)))
    plt.title('Plot ofthe angle part of FT of X(t)')
    plt.xlabel('Sample index')
    plt.ylabel('Angle')

def fourier_transform(sequence):
    """
    Apply FFT to a dataset ofsequence and return the magnitude 
    and angle parts.
    :param sequence: Design pattern sequence.
    """
    fxt = np.fft.fft(sequence)
    magnitude = np.abs(fxt)
    angle = np.angle(fxt)
    return magnitude, angle

def preprocessing(sequence):
    """
    Preprocess a dataset of sequence by applying fourier transform 
    then sigularize it. Although, we combine the magnitude and angle
    parts into a single array for later PCA analysis. It would still
    be a good idea to make them aligned in a nicer way
    :param sequence: Design pattern sequence.
    """
    magnitude, angle = fourier_transform(sequence)
    N = len(sequence)
    angle = N * angle % (2 * np.pi) / N
    result = np.concatenate([magnitude, angle], axis=1)
    return result
    
def simple_mapping(branch, trunk):
    """
    Define a simple functtion that maps index to PCA coordinates.
    :param branch: PCA coordinates.
    :param trunk: Index.
    """
    mask = np.arange(len(trunk)) % len(branch[0])
    branch = branch[0:,mask]
    return branch

def design_segementation(x, y, n, random_state=42):
    """
    Segment design patterns into n segments.
    Each segment has same length as the branch net input.
    :param x: Branch net input
    :param y: Design patterns.
    :param n: Number of segments for each design pattern.
    :param random_state: Random seed.
    """
    dataset = collections.defaultdict(list)
    segment_length = len(x[0])

    # Randomly sample segmenting position
    np.random.seed(random_state)
    index_array = np.random.randint(len(y[0]), size=(len(x), n))

    
    # segament design patterns
    for i in range(len(index_array)):
        for j in range(n):
            dataset['X'].append(x[i])
            mask = np.arange(index_array[i][j], index_array[i][j] + segment_length) % len(y[0])
            dataset['y'].append(y[i][mask])

    return dataset
            
def random_translation(y, random_state=42):
    """ 
    Translate design patters from a random postion
    :param y: Design patterns
    :param random_state: Random seed
    """
    translated_pattern = []
    num_row = len(y)
    num_col = len(y[0])
    # Randomly sample starting position
    np.random.seed(random_state)
    index_array = np.random.randint(1, num_col, size = num_row)

    # Translate dataset
    for i in range(num_row):
        mask = np.arange(index_array[i], index_array[i]+num_col) % num_col
        translated_pattern.append(y[i][mask])
    
    return np.array(translated_pattern)

def compare_test_pred(test, pred):
    """
    Visually compare model prediction with target value using bar plots.
    
    :param test: (1D array-like) Ground truth values.
    :param pred: (1D array-like) Model predictions.
    """
    indices = np.arange(len(test))  # X-axis indices

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, facecolor='lightblue')

    # Plot True Values (Top)
    axes[0].bar(indices, test, color='black', edgecolor='black')
    axes[0].set_ylabel("True Values")
    axes[0].set_title("Comparison of Model Predictions and True Values")

    # Plot Predictions (Bottom)
    axes[1].bar(indices, pred, color='red', edgecolor='black')
    axes[1].set_ylabel("Predictions")
    axes[1].set_xlabel("Index")

    # Adjust layout 
    plt.tight_layout()
    
    return fig

def quadratic(x):
    """
    A simple quadratic function for testing optimization algorithms.
    """
    return x @ x

def meent_tutorial(angle, efficiency, n_Si, pattern, period, polarization, thickness, wavelength):
    """
    Run EM silution using Meent
    """

    # Convert angle to radians for calculation
    angle_rad = np.radians(angle)

    # Assertion: check that period is equal to |wavelength / sin(angle)|
    if np.sin(angle_rad) != 0:  # avoid division by zero
        expected_period = abs(wavelength / np.sin(angle_rad))
        assert np.isclose(period, expected_period, rtol=1e-5), \
            f"Period {period} is not equal to |wavelength / sin(angle)| = {expected_period}"

    pol = 1 # TM polarized
    n_top = 1.45
    n_bot = 1.0
    theta = 0 * np.pi / 180
    phi = 0 * np.pi / 180
    thickness = [thickness]

    period = [period]
    fto = [40] 
    type_complex = np.complex128
    pattern = 0.5 * (pattern + 1)

    # Modeling
    ucell_1d_s = pattern.reshape((1, 1, len(pattern))) * (n_Si - 1) + 1
    mee = meent.call_mee(backend = 2,
                         pol = pol, n_top = n_top, n_bot = n_bot, theta = theta,
                         fto = fto, wavelength = wavelength, period = period,
                         ucell = ucell_1d_s, thickness = thickness,
                         type_complex = type_complex)
    result = mee.conv_solve()
    res = result.res
    de_ri = res.de_ri
    de_ti = res.de_ti
    ratio = wavelength/period
    if 0 <= ratio <= 1:
        de_angle = np.degrees(np.arcsin(ratio))
    else:
        print("Invalid input")
        de_angle = None
    return de_ti[0,fto[0]+1], de_angle

def meent_em(angle, n_Si, pattern, period, thickness, wavelength, backend=2):
    """
    Run EM silution using Meent
    """
    # backend 0 = Numpy
    # backend 1 = JAX
    # backend 2 = PyTorch

    # Convert angle to radians for calculation
    angle_rad = np.radians(angle)

    # Assertion: check that period is equal to |wavelength / sin(angle)|
    if np.sin(angle_rad) != 0:  # avoid division by zero
        expected_period = abs(wavelength / np.sin(angle_rad))
        assert np.isclose(period, expected_period, rtol=1e-5), \
            f"Period {period} is not equal to |wavelength / sin(angle)| = {expected_period}"

    pol = 1 # TM polarized
    n_top = 1.45
    n_bot = 1.0
    theta = 0 * np.pi / 180
    thickness = [thickness]
    period = [period]

    fto = [40] 
    type_complex = np.complex128

    if isinstance(pattern, (np.ndarray, torch.Tensor)):
        if pattern.ndim == 1:
            pattern_dim = pattern.shape[0]
        elif pattern.ndim == 2:
            pattern_dim = pattern.shape[1]
        else:
            raise ValueError("`pattern` must be a 1D or 2D array or tensor.")
    else:
        raise TypeError("`pattern` must be a NumPy ndarray or PyTorch tensor.")
    
    # Modeling
    ucell_1d_s = pattern.reshape((1, 1, pattern_dim)) * (n_Si - 1) + 1
    mee = meent.call_mee(backend = backend,
                         pol = pol, n_top = n_top, n_bot = n_bot, theta = theta,
                         fto = fto, wavelength = wavelength, period = period,
                         ucell = ucell_1d_s, thickness = thickness,
                         type_complex = type_complex)
    #mee.ucell.requires_grad = True
    result = mee.conv_solve()
    res = result.res
    de_ti = res.de_ti
    return de_ti[0,fto[0]+1]

def binary_cross_entropy(y_true, y_pred):
    """
    Compute binary cross-entropy loss.
    :param y_true: True labels.
    :param y_pred: Predicted labels.
    """
    y_pred = torch.clamp(y_pred, 1e-7, 1 - 1e-7)  # Clamping to avoid log(0)
    return -torch.mean(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))

def smooth_heaviside_func(x, beta, eta):
    numerator = torch.tanh(beta * eta) + torch.tanh(beta * (x - eta))
    denominator = torch.tanh(beta * eta) + torch.tanh(beta * (1 - eta))
    h = numerator / denominator
    return h

def sigmoid_np(x, beta, eta):
    return 1 / (1 + np.exp(-beta * (x - eta)))

def sigmoid_func(x, beta, eta):
    return torch.sigmoid(beta * (x - eta))

class DeepONetCartesianProdPlus(DeepONetCartesianProd):
    """Deep operator network for dataset in the format of Cartesian product.
    Where the output is added with a scaler the through an activation function.

    Args:
        layer_sizes_branch: A list of integers as the width of a fully connected network,
            or `(dim, f)` where `dim` is the input dimension and `f` is a network
            function. The width of the last layer in the branch and trunk net
            should be the same for all strategies except "split_branch" and "split_trunk".
        layer_sizes_trunk (list): A list of integers as the width of a fully connected
            network.
        activation: If `activation` is a ``string``, then the same activation is used in
            both trunk and branch nets. If `activation` is a ``dict``, then the trunk
            net uses the activation `activation["trunk"]`, and the branch net uses
            `activation["branch"]`.
        num_outputs (integer): Number of outputs. In case of multiple outputs, i.e., `num_outputs` > 1,
            `multi_output_strategy` below should be set.
        multi_output_strategy (str or None): ``None``, "independent", "split_both", "split_branch" or
            "split_trunk". It makes sense to set in case of multiple outputs.

            - None
            Classical implementation of DeepONet with a single output.
            Cannot be used with `num_outputs` > 1.

            - independent
            Use `num_outputs` independent DeepONets, and each DeepONet outputs only
            one function.

            - split_both
            Split the outputs of both the branch net and the trunk net into `num_outputs`
            groups, and then the kth group outputs the kth solution.

            - split_branch
            Split the branch net and share the trunk net. The width of the last layer
            in the branch net should be equal to the one in the trunk net multiplied
            by the number of outputs.

            - split_trunk
            Split the trunk net and share the branch net. The width of the last layer
            in the trunk net should be equal to the one in the branch net multiplied
            by the number of outputs.
        scaler_factor (int): multiple the output of DeepONet by a constant.
        bias_factor (int): minus a constant to the output of DeepONet.
        output_activation (str): the activation function for the output of DeepONet
    """
    def __init__(
        self,
        layer_sizes_branch,
        layer_sizes_trunk,
        activation,
        kernel_initializer,
        num_outputs=1,
        multi_output_strategy=None,
        scale_factor = 10,
        bias_factor = 0,
        output_activation = "sigmoid",
    ):
        super().__init__(
            layer_sizes_branch,
            layer_sizes_trunk,
            activation,
            kernel_initializer,
            num_outputs,
            multi_output_strategy,
        )
        self.register_buffer("scale_factor", torch.tensor(scale_factor, dtype=torch.float32))
        self.register_buffer("bias_factor", torch.tensor(bias_factor, dtype=torch.float32))
        if output_activation == "smooth_heaviside":
            self.output_activation = smooth_heaviside_func
        else:
            self.output_activation = sigmoid_func

    def forward(self, inputs):
        x = super().forward(inputs)
        return self.output_activation(x, self.scale_factor, self.bias_factor)
        

def inverse_fourier_transform(magnitude, angle):
    """
    Reconstruct the original sequence from its magnitude and angle
    using inverse FFT in PyTorch.
    
    :param magnitude: Magnitude of the Fourier Transform (1D torch.Tensor).
    :param angle: Phase angle of the Fourier Transform (1D torch.Tensor).
    :return: Reconstructed sequence (1D torch.Tensor, real part only).
    """
    # Reconstruct the complex spectrum
    real = magnitude * torch.cos(angle)
    imag = magnitude * torch.sin(angle)
    complex_spectrum = torch.complex(real, imag)

    # Apply inverse FFT
    reconstructed = torch.fft.ifft(complex_spectrum)

    # Return the real part of the reconstructed signal
    return reconstructed.real

def reconstruct_pca(PCA_coordinates, pca_components, pca_mean):
    """
    Reconstruct the original sequence from PCA coordinates.
    """
    recon_seq = PCA_coordinates @ pca_components + pca_mean
    recon_seq_mag = recon_seq[:, :256]
    recon_seq_ang = recon_seq[:, 256:]
    recon_seq_ang = (recon_seq_ang * 256) % (2 * np.pi)
    fxt = recon_seq_mag * torch.exp(1j * recon_seq_ang)
    recon_seq = torch.fft.ifft(fxt, dim = 1)
    return recon_seq.real

def BFGS(fun, x0, learning_rate=0.1, maxiter=1000,
         xtol=1e-8, ftol=1e-8, gtol=1e-8,
         callback=None):
    """
    Minimize a scalar function using BFGS with gradients computed by PyTorch.
    
    Parameters
    ----------
    fun : callable
        Function to be minimized. Takes a 1D torch.Tensor as input and returns a scalar torch.Tensor.
    x0 : torch.Tensor
        Initial guess. Must be a 1D tensor with requires_grad=True.
    learning_rate : float
        Step size scaling factor.
    maxiter : int
        Maximum number of iterations.
    xtol, ftol, gtol : float
        Convergence tolerances for x, f, and gradient norm respectively.
    callback : callable or None
        Optional callback function called as callback(xk) at each iteration.
    
    Returns
    -------
    x_best : torch.Tensor
        Estimated position of the minimum.
    info : dict
        Dictionary with details of the optimization process.
    """
    if x0.ndim != 1:
        raise ValueError("x0 must be a 1D tensor.")

    xk = x0.clone().detach().requires_grad_(True)
    n = xk.numel()
    Hk = torch.eye(n)

    fk = fun(xk)
    gk = torch.autograd.grad(fk, xk, create_graph=False)[0]

    x_best = xk.clone()
    f_best = fk.item()

    reason = "maxiter"  # Default reason for stopping
    for i in range(maxiter):
        if callback is not None:
            callback(xk)

        # Step direction
        pk = -Hk @ gk

        # Update x
        xnew = xk + learning_rate * pk
        xnew = xnew.detach().requires_grad_(True)

        fnew = fun(xnew)
        gnew = torch.autograd.grad(fnew, xnew, create_graph=False)[0]

        # BFGS update
        s = xnew - xk
        y = gnew - gk
        rho = s @ y

        if rho.item() < 1e-10:
            # Skip update to avoid instability
            print(f"Skipping update at iteration {i} due to small rho: {rho.item()}")
            reason = "small rho"
            break

        I = torch.eye(n)
        rho_inv = 1.0 / rho
        V = I - rho_inv * torch.outer(s, y)
        Hk = V @ Hk @ V.T + rho_inv * torch.outer(s, s)

        # Update best solution
        if fnew < f_best:
            x_best = xnew.clone()
            f_best = fnew.item()

        # Convergence checks
        if torch.norm(s) < xtol:
            reason = "xtol"
            break
        if torch.abs(fnew - fk) < ftol:
            reason = "ftol"
            break
        if torch.norm(gnew) < gtol:
            reason = "gtol"
            break
        # Prepare for next iteration
        xk = xnew
        fk = fnew
        gk = gnew

        if i % 10 == 0:
            print(f"Iteration {i}: f(x) = {fk.item()}, ||g(x)|| = {torch.norm(gk).item()}")

    return x_best.detach(), {
        "fun": fk.item(),
        "fun_best": f_best,
        "nit": i + 1,
        "grad_norm": torch.norm(gk).item(),
        "success": reason != "maxiter",
        "status": reason,
    }

class ObjectiveFunction:
    def __init__(self, fun, sigma, quad_points, args=()):
        self.fun = fun
        self.args = args
        self.sigma = sigma
        self.quad_points = quad_points
        self.dim = None
        self.basis = None
        self.gh_roots, self.gh_weights = np.polynomial.hermite.hermgauss(quad_points)

    def grad(self, x, *args):
        if self.dim is None:
            self.dim = len(x)
            self.basis = np.eye(self.dim)

        df_sigma_basis = np.zeros(self.dim)
        for d in range(self.dim):
            f_d = lambda t: self.fun(x + t * self.sigma * self.basis[d], *args)
            f_d_vals = np.array([f_d(p) for p in self.gh_roots])
            df_sigma_basis[d] = np.sum(
                self.gh_weights * self.gh_roots * f_d_vals
            ) / (self.sigma * np.sqrt(np.pi) / 2)
        return self.basis @ df_sigma_basis

    def __call__(self, x):
        return self.fun(x, *self.args), self.grad(x, *self.args)


def DGS_BFGS_function(fun, x0, sigma=0.1, quad_points=7, args=(),
             learning_rate=0.1, maxiter=1000,
             xtol=1e-8, ftol=1e-8, gtol=1e-8,
             callback=None, **options):
    """
    Minimize a scalar function using BFGS with smoothed directional gradients.

    Parameters
    ----------
    fun : callable
        Function to be minimized: fun(x, *args) -> float
    x0 : ndarray
        Initial guess.
    sigma : float
        Smoothing parameter for DGS.
    quad_points : int
        Number of Gauss-Hermite quadrature points.
    args : tuple
        Additional args for the objective.
    learning_rate : float
        Step size scaling factor.
    maxiter : int
        Max number of iterations.
    xtol, ftol, gtol : float
        Convergence tolerances.
    callback : callable
        Optional callback.
    options : dict
        Extra options (currently unused).

    Returns
    -------
    OptimizeResult
    """

    if x0.ndim != 1:
        raise ValueError("x0 must be a 1D array-like object.")

    # Wrap the function with DGS-style gradient estimation
    objective = ObjectiveFunction(fun, sigma=sigma, quad_points=quad_points, args=args)

    hist = dict()
    xk = x0.copy()
    x_best = xk.copy()
    f_best = np.inf
    fk, gfk = objective(xk)
    n = len(xk)
    Hk = np.eye(n)
    t = 0
    success = False

    hist['nfev'] = [t*quad_points*n]
    hist['f'] = [fk]
    for iteration in range(maxiter):
        pk = -Hk @ gfk
        sk = learning_rate * pk
        xnew = xk + sk
        fnew, gnew = objective(xnew)
        yk = gnew - gfk
        rho = 1.0 / (yk @ sk + 1e-12)

        I = np.eye(n)
        Vk = I - rho * np.outer(sk, yk)
        Hk = Vk @ Hk @ Vk.T + rho * np.outer(sk, sk)

        t += 1
        if callback is not None:
            callback(xnew)

        if fnew < f_best:
            x_best = xnew.copy()
            f_best = fnew

        # Convergence checks
        if np.linalg.norm(gnew, np.inf) < gtol:
            msg = 'Optimization terminated successfully (gtol).'
            success = True
            break
        if np.linalg.norm(xnew - xk, np.inf) < xtol:
            msg = 'Optimization terminated due to x-tolerance.'
            break
        if np.abs((fk - fnew) / (fk + 1e-8)) < ftol:
            msg = 'Optimization terminated due to f-tolerance.'
            break

        xk, fk, gfk = xnew, fnew, gnew
        hist['nfev'].append(t*quad_points*n)
        hist['f'].append(fk)

        if iteration % 10 == 0:
            print(f"Iteration {t}: f(x) = {fk}, ||g(x)|| = {np.linalg.norm(gfk)}")

    else:
        msg = 'Maximum number of iterations reached.'


    print("success:", success)
    print("message:", msg)
    print("final value:", fk)
    print("best value:", f_best)
    print("final x:", xk)
    print("best x:", x_best)
    print("iterations:", t)
    print("function evaluations:", t * quad_points * n)

    return x_best, OptimizeResult(x=xk, fun=fk, jac=gfk, nit=t, nfev = t*quad_points*n,
                                  x_best=x_best, f_best=f_best, hist=hist,
                          success=success, msg=msg)

class DGS_BFGS:
    def __init__(self):
        pass

    def from_unit_cube(self, x, lb, ub):
        assert lb.ndim == 1 and ub.ndim == 1 and x.ndim == 1
        return x * (ub - lb) + lb

    def RTfun_m(self, x, T, lt, dim, fun_m):
        assert x.ndim == 1
        return fun_m(np.matmul(x - lt, T))

    def DGS_grad(self, x, T, lt, A, rr, dim, GH_pts, gh_value_mat, gh_value_vec, gh_weight, fun_m):
        assert x.ndim == 1
        gh_value_mat_rot = np.matmul(gh_value_mat, A.T)
        rr_expand = np.repeat(rr, GH_pts)

        X_all_center = np.tile(x, (GH_pts * dim, 1)) + sqrt(2.0) * gh_value_mat_rot * rr_expand[:, None]
        X_all_center = X_all_center.squeeze()

        Y = np.zeros(GH_pts * dim)
        for i in range(GH_pts * dim):
            Y[i] = self.RTfun_m(X_all_center[i], T, lt, dim, fun_m)

        Y_weighted = Y * gh_value_vec
        Y_mat = np.reshape(Y_weighted, (GH_pts, dim), order='F')
        grad = np.matmul(gh_weight, Y_mat) * sqrt(2) / sqrt(pi) / rr
        grad = np.squeeze(np.matmul(A, grad[:, None]), axis=1)
        return grad, (GH_pts - 1) * dim

    def line_search(self, tau1, n1, tau2, n2):
        pts = np.zeros(n1 + n2)
        for i in range(len(pts)):
            if i < n1:
                pts[i] = tau1**i
            else:
                pts[i] = tau1**(n1 - 1) * tau2**(i + 1 - n1)
        return pts

    def find_min(self, fun_m, x0, GH_pts=7, bound=None, max_feval=5e5,
                 translation=False, rotation=False, pw=5.0, eps=1e-3,
                 res_step=20, tau1=0.9, tau2=0.4, n1=40, n2=20):
        """
        param fun_m: Callable function to minimize (must accept a 1D numpy array)
        param x0: Initial guess (1D numpy array)
        param GH_pts: Number of Gauss-Hermite quadrature points for DGS gradient estimation
        param bound: (unused) Placeholder for bounds - currently assumed [0,1]^d
        param max_feval: Maximum number of function evaluations
        param translation: If True, applies random translation to function landscape
        param rotation: If True, applies random rotation to function landscape
        param pw: Smoothing width multiplier (larger value = smoother gradient)
        param eps: Relative change threshold for adaptive radius resets
        param res_step: Window size for tracking step history and triggering resets
        param tau1: Decay factor for first stage of line search step distribution
        param tau2: Decay factor for second stage of line search step distribution
        param n1: Number of points with tau1 decay in line search
        param n2: Number of points with tau2 decay in line search
        """
        if x0.ndim != 1:
            raise ValueError("x0 must be a 1D numpy array")

        dim = x0.shape[0]
        num_ls = n1 + n2
        pts = self.line_search(tau1, n1, tau2, n2)

        lb = np.zeros(dim)
        ub = np.ones(dim)
        wi = 120

        gh = roots_hermite(GH_pts)
        gh_value = np.expand_dims(gh[0], axis=1)
        gh_weight = gh[1]
        gh_value_mat = np.zeros((GH_pts * dim, dim))
        gh_value_vec = np.zeros(GH_pts * dim)
        for i in range(dim):
            gh_value_mat[GH_pts * i:GH_pts * (i + 1), i] = gh[0]
            gh_value_vec[GH_pts * i:GH_pts * (i + 1)] = gh[0]

        T = np.eye(dim) if not rotation else special_ortho_group.rvs(dim)
        lt = np.zeros(dim) if not translation else self.from_unit_cube(np.random.rand(dim), lb + 0.1 * (ub - lb), ub - 0.1 * (ub - lb))

        xnew = x0.copy()
        fnew = self.RTfun_m(xnew, T, lt, dim, fun_m)

        f_value = [fnew]
        num_eval_array = [0]
        A = np.eye(dim)
        rr = pw * wi * np.ones(dim)
        maxl = sqrt(dim) * wi

        j = 0
        num_eval = 0
        step = 0
        upd = 0
        rel_upd = 0
        reset_id = 0
        x_hist = []

        grad_new, num = self.DGS_grad(xnew, T, lt, A, rr, dim, GH_pts, gh_value_mat, gh_value_vec, gh_weight, fun_m)
        num_eval += num
        Hk = np.eye(dim)

        while num_eval < max_feval:
            x_hist.insert(0, xnew)
            if len(x_hist) > res_step:
                x_hist.pop()
            dx_hist = np.linalg.norm(x_hist[0] - x_hist[-1]) if len(x_hist) >= 2 else 0

            if ((upd == 0) or (rel_upd < eps)) and (dx_hist < 0.05 * wi * sqrt(dim)) and (rr.mean() < 0.1 * wi) and (j == 0 or (j - reset_id >= res_step)):
                rr = random.gauss(pw * wi, 0.1 * pw * wi) * np.ones(dim)
                maxl = sqrt(dim) * wi
                reset_id = j
            else:
                rr = (dx_hist / sqrt(res_step) + rr) * 0.5
                maxl = 0.8 * maxl + 0.2 * dx_hist

            xold = xnew.copy()
            grad = grad_new.copy()
            d1 = -grad
            d2 = -Hk.dot(grad)

            beta1 = maxl / (np.linalg.norm(d1) + 1e-10)
            beta2 = maxl / (np.linalg.norm(d2) + 1e-10)

            fl1 = []
            xl1 = []
            for pt in pts:
                x_try = xold + beta1 * d1 * pt
                xl1.append(x_try)
                fl1.append(self.RTfun_m(x_try, T, lt, dim, fun_m))
                num_eval += 1

            fl2 = []
            xl2 = []
            for pt in pts:
                x_try = xold + beta2 * d2 * pt
                xl2.append(x_try)
                fl2.append(self.RTfun_m(x_try, T, lt, dim, fun_m))
                num_eval += 1

            fl1 = np.array(fl1)
            fl2 = np.array(fl2)

            if fl2.min() < fl1.min():
                idx = fl2.argmin()
                xnew = xl2[idx]
                fnew = fl2[idx]
                rule_eff = "bfgs"
            else:
                idx = fl1.argmin()
                xnew = xl1[idx]
                fnew = fl1[idx]
                rule_eff = "standard"

            grad_new, num = self.DGS_grad(xnew, T, lt, A, rr, dim, GH_pts, gh_value_mat, gh_value_vec, gh_weight, fun_m)
            num_eval += num

            sk = xnew - xold
            yk = grad_new - grad
            denom = yk.dot(sk)
            if denom > 1e-10:
                rho_k = 1.0 / denom
                Hk = (np.eye(dim) - rho_k * np.outer(sk, yk)).dot(Hk).dot(np.eye(dim) - rho_k * np.outer(yk, sk)) + rho_k * np.outer(sk, sk)

            num_eval_array.append(num_eval)
            f_value.append(fnew)

            step = np.linalg.norm(xnew - xold)
            upd = abs(f_value[-1] - f_value[-2])
            rel_upd = upd / (abs(f_value[-2]) + 1e-12)

            if j % 1 == 0:
                print(f"Iter {j + 1}; feval {num_eval}; loss = {fnew:.2e}; norm_grad = {np.linalg.norm(grad):.2e}; dx_hist = {dx_hist:.2e}; rr = {rr[0]:.2e}; maxl = {maxl:.2e}; cond = {np.linalg.cond(Hk):.2e}; rule = {rule_eff}")

            j += 1
            if fnew < 1e-8:
                break

        print("Optimization finished.")
        return xnew, fnew, np.array(num_eval_array), np.array(f_value)
    
class Hybrid_DGS_BFGS:
    def __init__(self):
        pass

    def from_unit_cube(self, x, lb, ub):
        assert lb.ndim == 1 and ub.ndim == 1 and x.ndim == 1
        return x * (ub - lb) + lb

    def create_block_matrix(self, v, m, d):
      # Creates a block diagonal matrix with `d` blocks of shape (m,1)
        blocks = [v for _ in range(d)]  # list of d blocks
        block_diag = torch.block_diag(*blocks)  # shape (m*d, d)
        return block_diag.T

    def RTfun_m(self, x, T, lt, fun_m, fun_s):
        x = torch.as_tensor(x, dtype=torch.float32)
        return fun_s(fun_m((x - lt) @ T))

    def DGS_Jacobian(self, x, fun_m, gh_value_vec, gh_weight, sigma, A):
        m = self.GH_pts
        d = self.x_dim
        c = torch.sqrt(torch.tensor(2.0)) / (torch.sqrt(torch.tensor(np.pi)) * sigma)
        sqrt2  = torch.sqrt(torch.tensor(2.0, device=x.device, dtype=x.dtype))
        # perturbed x for every GH points (m, 25, 25)
        v = gh_value_vec.view(m, 1, 1) * A.t().view(1, d, d)
        perturbs = x.view(1, 1, d) + sqrt2 * sigma * v
        perturbs = perturbs.reshape(-1, d)
        # (m , 25, 256)
        f_vals = fun_m(perturbs)
        f_vals = f_vals.reshape(m, d, -1)
        # (25, 256)
        Jacobian_estimate = c * ((gh_weight * gh_value_vec)[:, None, None] * f_vals).sum(dim = 0)
        return Jacobian_estimate.T

    def hybrid_DGS_grad(self, x, fun_s, fun_m, gh_value_vec, gh_weight, sigma, A):
        x = x.detach().requires_grad_(True)
        jac_dgs = self.DGS_Jacobian(x, fun_m, gh_value_vec, gh_weight, sigma, A)  # (output_dim, input_dim)
        y = fun_m(x)  # output vector
        y.retain_grad()
        loss = fun_s(y)
        loss.backward()
        return y.grad @ jac_dgs, 1  # gradient vector and feval=1

    def line_search(self, tau1, n1, tau2, n2):
        pts = torch.zeros(n1 + n2)
        for i in range(n1 + n2):
            if i < n1:
                pts[i] = tau1 ** i
            else:
                pts[i] = tau1 ** (n1 - 1) * tau2 ** (i + 1 - n1)
        return pts

    def find_min(self, fun_m, fun_s, x0, GH_pts=7, max_feval=5e5,
                 min_df = 1e-8,
                 translation=False, rotation=False, pw=5.0, eps=1e-3,
                 res_step=20, tau1=0.9, tau2=0.4, n1=40, n2=20):

        if x0.ndim != 1:
            raise ValueError("x0 must be a 1D tensor")
        
        self.GH_pts = GH_pts
        self.x_dim = x0.shape[0]

        dim = x0.shape[0]
        num_ls = n1 + n2
        pts = self.line_search(tau1, n1, tau2, n2)
        device = x0.device
        dtype = x0.dtype

        lb = torch.zeros(dim, device=device)
        ub = torch.ones(dim, device=device)
        wi = 120

        gh = roots_hermite(GH_pts)
        gh_value_vec = torch.tensor(gh[0], dtype=dtype, device=device)
        gh_weight = torch.tensor(gh[1], dtype=dtype, device=device)

        T = torch.eye(dim, device=device) if not rotation else torch.tensor(special_ortho_group.rvs(dim), dtype=dtype, device=device)
        lt = torch.zeros(dim, device=device) if not translation else self.from_unit_cube(torch.rand(dim, device=device), lb + 0.1 * (ub - lb), ub - 0.1 * (ub - lb))

        A = torch.eye(dim, device=device)
        rr = pw * wi * torch.ones(dim, device=device)
        maxl = math.sqrt(dim) * wi

        xnew = x0.clone()
        fnew = self.RTfun_m(xnew, T, lt, fun_m, fun_s)

        f_value = [fnew.item()]
        num_eval_array = [0]
        x_hist = []

        grad_new, num = self.hybrid_DGS_grad(xnew, fun_s, lambda x: fun_m((x - lt) @ T), gh_value_vec, gh_weight, rr.mean(), A)
        num_eval = num
        Hk = torch.eye(dim, device=device)

        j = 0
        upd = 0
        rel_upd = 0
        reset_id = 0

        while num_eval < max_feval:
            x_hist.insert(0, xnew.clone())
            if len(x_hist) > res_step:
                x_hist.pop()
            dx_hist = torch.norm(x_hist[0] - x_hist[-1]) if len(x_hist) >= 2 else 0

            if ((upd == 0 or rel_upd < eps) and dx_hist < 0.05 * wi * math.sqrt(dim) and rr.mean() < 0.1 * wi and (j == 0 or (j - reset_id >= res_step))):
                rr = torch.randn(1).abs() * 0.1 * pw * wi + pw * wi
                rr = rr * torch.ones(dim, device=device)
                maxl = math.sqrt(dim) * wi
                reset_id = j
            else:
                rr = (dx_hist / math.sqrt(res_step) + rr) * 0.5
                maxl = 0.8 * maxl + 0.2 * dx_hist

            xold = xnew.clone()
            grad = grad_new.clone()
            d1 = -grad
            d2 = -grad @ Hk.T

            beta1 = maxl / (torch.norm(d1) + 1e-10)
            beta2 = maxl / (torch.norm(d2) + 1e-10)

            fl1 = []
            xl1 = []
            fl2 = []
            xl2 = []
            with torch.no_grad():
              for pt in pts:
                  x_try = xold + beta1 * d1 * pt
                  xl1.append(x_try)
                  fl1.append(self.RTfun_m(x_try, T, lt, fun_m, fun_s))
                  num_eval += 1

        
              for pt in pts:
                  x_try = xold + beta2 * d2 * pt
                  xl2.append(x_try)
                  fl2.append(self.RTfun_m(x_try, T, lt, fun_m, fun_s))
                  num_eval += 1

              fl1 = torch.stack(fl1)
              fl2 = torch.stack(fl2)

              if fl2.min() < fl1.min():
                  idx = fl2.argmin()
                  xnew = xl2[idx]
                  fnew = fl2[idx]
                  rule_eff = "bfgs"
              else:
                  idx = fl1.argmin()
                  xnew = xl1[idx]
                  fnew = fl1[idx]
                  rule_eff = "standard"

            grad_new, num = self.hybrid_DGS_grad(xnew, fun_s, lambda x: fun_m((x - lt) @ T), gh_value_vec, gh_weight, rr.mean(), A)
            num_eval += num

            sk = (xnew - xold).squeeze()
            yk = (grad_new - grad).squeeze()
            denom = yk @ sk
            if denom > 1e-10:
                rho = 1.0 / denom
                I = torch.eye(dim, device=device)
                Hk = (I - rho * torch.outer(sk, yk)) @ Hk @ (I - rho * torch.outer(yk, sk)) + rho * torch.outer(sk, sk)

            num_eval_array.append(num_eval)
            f_value.append(fnew.item())

            step = torch.norm(xnew - xold)
            upd = abs(f_value[-1] - f_value[-2])
            rel_upd = upd / (abs(f_value[-2]) + 1e-12)

            print(f"Iter {j + 1}; feval {num_eval}; loss = {fnew.item():.2e}; "
                  f"||grad|| = {torch.norm(grad):.2e}; dx_hist = {dx_hist:.2e}; "
                  f"rr = {rr[0].item():.2e}; maxl = {maxl:.2e}; rule = {rule_eff}")

            j += 1
            if fnew.item() < 1e-8:
                break

            # early stop when the change of function value is too small
            if j > 1 and abs(f_value[-1] - f_value[-2]) < min_df:
                print("Early stopping due to small change in function value.")
                break

        print("Optimization finished.")
        return xnew.detach(), fnew.item(), np.array(num_eval_array), np.array(f_value)
    
def sign_dist(x, rho, eta, func, res):
    """
    Vectorized calculation of the signed distance.

    Parameters:
    x    : numpy array, shape (n_points, n_features)
    rho  : latent coordinates (n_samples, n_features)
    eta  : float
    func    : callable, maps (n_points, n_features) X (n_samples, n_features) -> (n_samples, n_points)
    res : distance for finite difference approximation

    Returns:
    d    : numpay array, shape (n_samples,)
    """
    y = func((rho, x))  # (n_samples, n_points)
    
    # Compute gradient using finite difference approximation
    y_pre = func((rho, x - res))  # (n_samples, n_points)
    y_post = func((rho, x + res))  # (n_samples, n_points)
    grad = (y_post - y_pre) / (2 * res)  # (n_samples, n_points)

    # Return signed distance
    return (eta - y) / torch.abs(grad + 1e-8) # (n_samples, n_points)

def subpixel_projection(y, res):
    """
    y: Linearized approximiation of signed distance (n_samples, seq_len)
    res: Resolution threshold for subpixel smoothing

    return: torch.Tensor (n_samples, seq_len)
    """
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32, device=device)
    
    y_norm = y / res
    
    # Compute smoothed values
    smoothed = (
        0.5 - (15 / 16) * y_norm
        + (5 / 8) * y_norm**3
        - (3 / 16) * y_norm**5
    )

    # Apply conditions element-wise using nested torch.where
    result = torch.where(
        y < -res,
        torch.ones_like(y),  # y < -res → 1
        torch.where(
            y > res,
            torch.zeros_like(y),  # y > res → 0
            smoothed  # -res <= y <= res → smoothed transition
        )
    )

    return result
def rastrigin(x):
    """
    Rastrigin function, a common test problem for optimization algorithms.
    :param x: Input array.
    :return: Function value.
    """
    A = 10
    assert x.ndim == 1
    n = len(x)
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def rastrigin_torch(x):
    """
    Rastrigin function implemented in PyTorch.
    :param x: Input tensor.
    :return: Function value.
    """
    A = 10
    assert x.ndim == 1
    n = x.shape[0]
    return A * n + torch.sum(x**2 - A * torch.cos(2 * np.pi * x))



if __name__ == "__main__":
    if torch.cuda.is_available():
        device = 'cuda'

    optimizer = Hybrid_DGS_BFGS()
    x0 = torch.tensor([3]*25, device=device, dtype=torch.float32).requires_grad_(True)
      # Initial guess for the optimization
    x_opt, f_opt, num_eval_array, f_value = optimizer.find_min(
        lambda x: x,
        rastrigin_torch,
        x0,
        GH_pts = 7,
    )
    print("Optimal x:", x_opt)
    print("Optimal f(x):", f_opt)

    

