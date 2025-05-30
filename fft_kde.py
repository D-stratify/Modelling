
import numpy as np

from scipy.fft import fft, ifft
from scipy.signal import fftconvolve
from numba import njit

from scipy.fft import fft2, ifft2
from scipy.interpolate import RegularGridInterpolator
#from scipy.interpolate import interp1d

# Kernel functions using Numba for performance

@njit
def K_h_numba(x, h):
    coeff = 1.0 / (h * np.sqrt(2 * np.pi))
    return np.exp(-0.5 * (x / h)**2) * coeff

@njit
def K_hp_numba(x, h, L):
    n = len(x)
    result = np.empty(n)
    coeff = 1.0 / (np.sqrt(2 * np.pi) * h)
    for i in range(n):
        total = 0.0
        for k in range(-1, 2):  # k in [-3, ..., 3]
            diff = x[i] + k * L
            total += np.exp(-0.5 * (diff / h) ** 2)
        result[i] = total * coeff
    return result

@njit
def dK_hp_numba(x, h, L):
    n = len(x)
    result = np.empty(n)
    coeff = 1.0 / (np.sqrt(2 * np.pi) * h ** 3)
    for i in range(n):
        total = 0.0
        for k in range(-1, 2):
            diff = x[i] + k * L
            total += -diff * np.exp(-0.5 * (diff / h) ** 2)
        result[i] = total * coeff
    return result

# 2D kernel construction using Numba for performance

def build_2d_kernel(x_grid, y_grid, hx, hy, L):
    # Wrapped Gaussian in x
    
    #Kx = wrapped_gaussian(x_grid, hx, L)
    Kx = K_hp_numba(x=x_grid, h=hx, L=L)
    Kx /= Kx.sum()

    # Standard Gaussian in y
    #Ky = np.exp(-0.5 * (y_grid / hy) ** 2) / (hy * np.sqrt(2 * np.pi))
    Ky = K_h_numba(x=y_grid, h=hy)
    Ky /= Ky.sum()

    # Outer product: 2D kernel
    K = np.outer(Kx, Ky)
    
    return np.roll(K, shift=(-len(x_grid)//2, -len(y_grid)//2), axis=(0,1))  # center the kernel

def build_2d_gaussian_kernel(x_grid, y_grid, hx, hy):
    Kx = K_h_numba(x=x_grid, h=hx)
    Ky = K_h_numba(x=y_grid, h=hy)

    Kx /= Kx.sum()
    Ky /= Ky.sum()

    return np.outer(Kx, Ky)


# FFT-based 1D KDE with support for standard or periodic (wrapped) kernels

def fft_kde(X_points, x_grid, h=0.1, kernel_type='gaussian', L=1.0):
    """
    FFT-based 1D KDE using either Gaussian or wrapped (periodic) kernel.

    Parameters:
    X_points : ndarray
        1D data points.
    x_grid : ndarray
        Uniform evaluation grid.
    h : float
        Bandwidth.
    kernel_type : str
        'gaussian' for standard KDE, 'wrapped' for periodic KDE.
    L : float
        Period length for wrapped kernel.

    Returns:
    f_X : ndarray
        Density estimate evaluated at x_grid.
    """
    N = len(X_points)
    M = len(x_grid)
    x_min, x_max = x_grid[0], x_grid[-1]
    dx = x_grid[1] - x_grid[0]

    # Histogram the data (non-normalized counts)
    hist, _ = np.histogram(X_points, bins=M, range=(x_min, x_max))

    # Grid for kernel
    kernel_grid = np.linspace(-M//2 * dx, M//2 * dx, M)

    if kernel_type == 'gaussian':
        kernel_vals = K_h_numba(kernel_grid, h)
        kernel_vals /= kernel_vals.sum()
        # Use zero-padded convolution for non-periodic
        density = fftconvolve(hist, kernel_vals, mode='same')
    elif kernel_type == 'wrapped':
        kernel_vals = K_hp_numba(kernel_grid, h, L=L)
        kernel_vals /= kernel_vals.sum()
        # Circular convolution via FFT for periodic KDE
        kernel_vals = np.roll(kernel_vals, -M // 2)  # Center kernel
        density = np.real(ifft(fft(hist) * fft(kernel_vals)))
    else:
        raise ValueError("kernel_type must be 'gaussian' or 'wrapped'.")

    # Normalize to return density (PDF)
    f_X = density / (N * dx)
    return f_X


# FFT-based 2D KDE with  support for standard or periodic (wrapped) kernels

def fft_kde_2d(X, Y, x_grid, y_grid, hx=0.1, hy=0.1):
    """
    2D KDE using FFT convolution (non-periodic in both X and Y).
    
    Parameters:
        X, Y       : Input samples (1D arrays)
        x_grid     : Uniform grid for X (length Mx)
        y_grid     : Uniform grid for Y (length My)
        hx, hy     : Bandwidths for X and Y

    Returns:
        f_xy       : 2D array of shape (Mx, My) with KDE estimate
    """
    Mx, My = len(x_grid), len(y_grid)
    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]

    # Histogram over grid range
    hist2d, _, _ = np.histogram2d(X, Y, bins=[Mx, My],
                                  range=[[x_grid[0], x_grid[-1]], [y_grid[0], y_grid[-1]]])

    # Kernel grid
    x_kernel = np.linspace(-Mx // 2 * dx, Mx // 2 * dx, Mx)
    y_kernel = np.linspace(-My // 2 * dy, My // 2 * dy, My)

    # 2D Gaussian kernel
    kernel = build_2d_gaussian_kernel(x_kernel, y_kernel, hx, hy)

    # Perform full 2D convolution
    density = fftconvolve(hist2d, kernel, mode='same')

    # Normalize to obtain PDF
    f_xy = density / (len(X) * dx * dy)

    return f_xy

def fft_kde_2d_mixed(X, Y, x_grid, y_grid, hx=0.1, hy=0.1, L=1.0):
    """
    2D KDE where X is periodic and Y is not.
    
    Parameters:`
        X, Y       : input samples (1D arrays)
        x_grid     : uniform grid for periodic X (length Mx)
        y_grid     : uniform grid for non-periodic Y (length My)
        hx, hy     : bandwidths in X and Y
        L          : period for X

    Returns:
        f_xy       : 2D array of shape (Mx, My) with KDE estimate
    """
    Mx, My = len(x_grid), len(y_grid)
    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]

    # 2D histogram
    hist2d, _, _ = np.histogram2d(X, Y, bins=[Mx, My], range=[[0, L], [y_grid[0], y_grid[-1]]])

    # Centered grid for kernel
    x_kernel = np.linspace(-Mx // 2 * dx, Mx // 2 * dx, Mx)
    y_kernel = np.linspace(-My // 2 * dy, My // 2 * dy, My)

    # 2D kernel (wrapped in x)
    kernel = build_2d_kernel(x_kernel, y_kernel, hx, hy, L)

    # FFT-based 2D convolution (periodic in x, linear in y)
    density = np.real(ifft2(fft2(hist2d) * fft2(kernel)))

    # Normalize to get PDF
    f_xy = density / (len(X) * dx * dy)

    return f_xy



# Weighted 1D/2D KDE using FFT convolution, with support for periodic (wrapped) kernels

def fft_kde_weighted(X, weights, x_grid, h=0.1, kernel_type='gaussian', L=1.0):
    """
    Weighted 1D KDE using FFT convolution, with support for periodic (wrapped) kernels.

    Parameters:
    X : ndarray
        Input data points.
    weights : ndarray
        Weights (e.g., function values or conditional numerators).
    x_grid : ndarray
        Uniform grid to evaluate KDE on.
    h : float
        Bandwidth.
    kernel_type : str
        'gaussian' or 'wrapped'.
    L : float
        Period length (used only for 'wrapped' kernel).

    Returns:
    kde_values : ndarray
        KDE estimate evaluated at x_grid.
    """
    N = len(X)
    M = len(x_grid)
    x_min, x_max = x_grid[0], x_grid[-1]
    dx = x_grid[1] - x_grid[0]

    # Step 1: Histogram the weighted data
    hist, _ = np.histogram(X, bins=M, range=(x_min, x_max), weights=weights)

    # Step 2: Build kernel grid
    kernel_grid = np.linspace(-M // 2 * dx, M // 2 * dx, M)

    if kernel_type == 'gaussian':
        kernel_vals = K_h_numba(kernel_grid, h)
        kernel_vals /= kernel_vals.sum()

        # Standard (zero-padded) convolution
        smoothed = fftconvolve(hist, kernel_vals, mode='same')

    elif kernel_type == 'wrapped':
        kernel_vals = K_hp_numba(kernel_grid, h, L=L)
        kernel_vals /= kernel_vals.sum()

        # Circular convolution via FFT
        kernel_vals = np.roll(kernel_vals, -M // 2)
        smoothed = np.real(ifft(fft(hist) * fft(kernel_vals)))

    else:
        raise ValueError("kernel_type must be 'gaussian' or 'wrapped'.")

    return smoothed / dx

def fft_kde_derivative(X, x_grid, h=0.1, weights=None, kernel_type='wrapped', L=1.0):
    N = len(X)
    M = len(x_grid)
    dx = x_grid[1] - x_grid[0]
    x_min, x_max = x_grid[0], x_grid[-1]

    if weights is None:
        weights = np.ones_like(X)

    hist, _ = np.histogram(X, bins=M, range=(x_min, x_max), weights=weights)
    kernel_grid = np.linspace(-M // 2 * dx, M // 2 * dx, M)

    if kernel_type == 'wrapped':
        dkernel_vals = dK_hp_numba(kernel_grid, h, L)
        dkernel_vals -= dkernel_vals.mean()  # Ensure kernel integrates to 0
        dkernel_vals = np.roll(dkernel_vals, -M // 2)
        smoothed = np.real(ifft(fft(hist) * fft(dkernel_vals)))
    else:
        raise ValueError("Only 'wrapped' kernel supported in derivative.")

    return smoothed / dx

def conditional_mean_1D_fft(Y, X, x_grid, h=0.1, kernel_type='gaussian', L=1.0):
    """
    Fast conditional mean E[Y | X = x] using FFT-based KDE with interpolation.
    
    Parameters:
    Y, X : ndarray
        Data arrays.
    x_grid : ndarray
        Points to evaluate conditional expectation on (uniform grid).
    h : float
        Kernel bandwidth.

    Returns:
    cond_mean : ndarray
        E[Y | X = x_grid] estimates.
    interpolator : function
        Function to evaluate E[Y | X = x] at arbitrary x points.
    """
    # Numerator: weighted KDE with weights = Y
    numerator = fft_kde_weighted(X, Y, x_grid, h, kernel_type, L)
    
    # Denominator: standard KDE
    denominator = fft_kde_weighted(X, np.ones_like(Y), x_grid, h, kernel_type, L)

    # Avoid divide-by-zero
    with np.errstate(divide='ignore', invalid='ignore'):
        cond_mean = np.where(denominator > 1e-12, numerator / denominator, 0.0)

    
    # Interpolator for arbitrary x points (handle periodicity)
    def periodic_interpolator(x_query):
        if kernel_type == 'wrapped':
            x_query = np.asarray(x_query) % L  # enforce periodicity    
        return RegularGridInterpolator((x_grid,), cond_mean, bounds_error=False, fill_value=0.0)(x_query)
    interpolator = periodic_interpolator
    
    return cond_mean, interpolator

def conditional_mean_2D_fft(Z, X, Y, x_grid, y_grid, hx=0.05, hy=0.2, L=1.0):
    """
    Estimate E[Z | X = x, Y = y] where X is periodic and Y is not.

    Parameters:
        Z: values to predict (N,)
        X: periodic input (N,)
        Y: real-valued input (N,)
        x_grid: grid for X (length Mx)
        y_grid: grid for Y (length My)

    Returns:
        cond_E: 2D array of E[Z|x,y] on (x_grid, y_grid)
        interpolator: function (x, y) -> E[Z|x,y] using bilinear interpolation
    """
    Mx, My = len(x_grid), len(y_grid)
    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]
    N = len(X)

    # 2D histograms
    hist_unweighted, _, _ = np.histogram2d(X, Y, bins=[Mx, My],
                                           range=[[0, L], [y_grid[0], y_grid[-1]]])

    hist_weighted, _, _ = np.histogram2d(X, Y, bins=[Mx, My],
                                         range=[[0, L], [y_grid[0], y_grid[-1]]],
                                         weights=Z)

    # Kernel grid
    x_kernel = np.linspace(-Mx//2 * dx, Mx//2 * dx, Mx)
    y_kernel = np.linspace(-My//2 * dy, My//2 * dy, My)
    kernel = build_2d_kernel(x_kernel, y_kernel, hx, hy, L)

    # FFT convolution
    kernel_fft = fft2(kernel)
    smooth_unweighted = np.real(ifft2(fft2(hist_unweighted) * kernel_fft))
    smooth_weighted   = np.real(ifft2(fft2(hist_weighted  ) * kernel_fft))

    # Conditional expectation
    with np.errstate(divide='ignore', invalid='ignore'):
        cond_E = np.where(smooth_unweighted > 1e-12,
                          smooth_weighted / smooth_unweighted,
                          0.0)

    # Interpolator (handle periodic x by reducing modulo L in queries)
    interpolator = RegularGridInterpolator((x_grid, y_grid), cond_E.T,
                                           bounds_error=False, fill_value=0.0)

    def conditional_interp(x_query, y_query):
        x_query = np.asarray(x_query) % L  # enforce periodicity
        y_query = np.asarray(y_query)
        points = np.stack([x_query, y_query], axis=-1)
        return interpolator(points)

    return cond_E, conditional_interp


# Derivative of conditional 1D mean using FFT-based KDE

def fft_kde_weighted_test(X, weights, x_grid, h=0.05, L=1.0):
    M = len(x_grid)
    dx = x_grid[1] - x_grid[0]

    bin_edges = np.linspace(0, L, M + 1)
    hist, _ = np.histogram(X, bins=bin_edges, weights=weights)

    # Proper symmetric spatial kernel grid
    kernel_grid = (np.arange(M) - M // 2) * dx
    kernel_vals = K_hp_numba(kernel_grid, h, L)
    kernel_vals /= np.sum(kernel_vals)

    kernel_vals = np.roll(kernel_vals, -M // 2)
    smoothed = np.real(ifft(fft(hist) * fft(kernel_vals)))

    return smoothed / dx

def fft_kde_derivative_test(X, x_grid, h=0.1, weights=None, L=1.0):
    M = len(x_grid)
    dx = x_grid[1] - x_grid[0]

    if weights is None:
        weights = np.ones_like(X)

    bin_edges = np.linspace(0, L, M + 1)
    hist, _ = np.histogram(X, bins=bin_edges, weights=weights)

    kernel_grid = (np.arange(M) - M // 2) * dx
    dkernel_vals = dK_hp_numba(kernel_grid, h, L)
    dkernel_vals -= np.mean(dkernel_vals)
    dkernel_vals = np.roll(dkernel_vals, -M // 2)

    smoothed = np.real(ifft(fft(hist) * fft(dkernel_vals)))

    return smoothed #/ dx

def conditional_mean_1D_derivative_fft(Y, X, x_grid, h=0.05, L=1.0):
    g = fft_kde_weighted_test(X, Y, x_grid, h=h, L=L)
    f_X = fft_kde_weighted_test(X, np.ones_like(Y), x_grid, h=h, L=L)

    g_prime = fft_kde_derivative_test(X, x_grid, h=h, weights=Y, L=L)
    f_X_prime = fft_kde_derivative_test(X, x_grid, h=h, weights=np.ones_like(X), L=L)

    with np.errstate(divide='ignore', invalid='ignore'):
        dEdx = (f_X * g_prime - g * f_X_prime) / (f_X ** 2)
        dEdx[~np.isfinite(dEdx)] = 0.0

    
    # Interpolator for arbitrary x points (handle periodicity)
    def periodic_interpolator(x_query):
        x_query = np.asarray(x_query) % L  # enforce periodicity
        return RegularGridInterpolator((x_grid,), dEdx, bounds_error=False, fill_value=0.0)(x_query)
    interpolator = periodic_interpolator
    
    return dEdx, interpolator
