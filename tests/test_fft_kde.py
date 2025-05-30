
"""
Test cases for FFT-based kernel density estimation (KDE) and conditional mean estimation.

To run these tests, ensure you have pytest installed and run:

    python3 -m pytest tests/

from the root directory of the project.
"""

import numpy as np
from scipy.stats import norm

import sys
import os
sys.path.append(os.path.abspath("../"))

# Import the function under test
from fft_kde import *
from scipy.stats import multivariate_normal


# Test cases for the 1D FFT-based kernel density estimation (KDE)

def test_fft_kde_gaussian_density():
    # Test fft_kde with Gaussian kernel on normal data
    np.random.seed(0)
    N = 1000
    X = np.random.normal(0, 1, N)
    x_grid = np.linspace(-4, 4, 200)
    h = 0.02

    f_X = fft_kde(X, x_grid, h=h, kernel_type='gaussian')
    # Compare to true normal PDF
    true_pdf = norm.pdf(x_grid, 0, 1)
    mae = np.mean(np.abs(f_X - true_pdf))
    assert mae < 0.05, f"Mean absolute error too large for gaussian fft_kde: {mae}"

def test_fft_kde_wrapped_density():
    # Test fft_kde with wrapped kernel on uniform periodic data
    np.random.seed(1)
    N = 1000
    L = 2 * np.pi
    X = np.random.uniform(0, L, N)
    x_grid = np.linspace(0, L, 200)
    h = 0.02

    f_X = fft_kde(X, x_grid, h=h, kernel_type='wrapped', L=L)
    # True PDF is uniform
    true_pdf = np.ones_like(x_grid) / L
    mae = np.mean(np.abs(f_X - true_pdf))
    assert mae < 0.05, f"Mean absolute error too large for wrapped fft_kde: {mae}"

def test_fft_kde_invalid_kernel_type():
    # Test fft_kde raises ValueError for invalid kernel_type
    X = np.random.normal(0, 1, 100)
    x_grid = np.linspace(-3, 3, 100)
    try:
        fft_kde(X, x_grid, h=0.1, kernel_type='invalid')
    except ValueError as e:
        assert "kernel_type" in str(e)
    else:
        assert False, "fft_kde did not raise ValueError for invalid kernel_type"


# Test cases for the 2D FFT-based kernel density estimation (KDE)

def test_fft_kde_2d_gaussian_density():
    # Test fft_kde_2d with 2D Gaussian data
    np.random.seed(123)
    N = 100
    mean = [0.0, 0.0]
    cov = [[1.0, 0.5], [0.5, 1.5]]
    X, Y = np.random.multivariate_normal(mean, cov, N).T

    x_grid = np.linspace(-4, 4, 100)
    y_grid = np.linspace(-5, 5, 120)
    hx, hy = 0.2, 0.3

    f_xy = fft_kde_2d(X, Y, x_grid, y_grid, hx=hx, hy=hy)

    # True PDF at grid points
    Xg, Yg = np.meshgrid(x_grid, y_grid, indexing='ij')
    pos = np.stack([Xg, Yg], axis=-1)
    true_pdf = multivariate_normal(mean, cov).pdf(pos)

    # Normalize both for comparison (since KDE is approximate)
    f_xy /= f_xy.sum()
    true_pdf /= true_pdf.sum()

    mae = np.mean(np.abs(f_xy - true_pdf))
    assert mae < 0.05, f"Mean absolute error too large for 2D fft_kde: {mae}"

def test_fft_kde_2d_mixed_periodic_x():
    # Test fft_kde_2d_mixed with X periodic and Y non-periodic (Gaussian in Y)
    np.random.seed(2024)
    N = 500
    L = 2 * np.pi
    # X is uniform on [0, L), Y is normal
    X = np.random.uniform(0, L, N)
    Y = np.random.normal(0, 1, N)
    x_grid = np.linspace(0, L, 100)
    y_grid = np.linspace(-4, 4, 120)
    hx, hy = 0.15, 0.25

    # Estimate density
    f_xy = fft_kde_2d_mixed(X, Y, x_grid, y_grid, hx=hx, hy=hy, L=L)

    # True PDF: uniform in X, normal in Y
    true_pdf = np.ones((len(x_grid), len(y_grid))) / L * norm.pdf(y_grid[None, :], 0, 1)

    # Normalize both for comparison
    f_xy /= f_xy.sum()
    true_pdf /= true_pdf.sum()

    mae = np.mean(np.abs(f_xy - true_pdf))
    assert mae < 0.05, f"Mean absolute error too large for fft_kde_2d_mixed: {mae}"


# Test cases for the conditional mean and its derivative using FFT-based KDE

def test_conditional_mean_1D_fft():
    # Generate synthetic data: Y = 2*X + 1 + noise
    np.random.seed(42)
    N = 1000
    X = np.random.uniform(-2, 2, N)
    Y = 2 * X + 1 + np.random.normal(0, 0.1, N)
    x_grid = np.linspace(-2, 2, 200)
    h = 0.02

    # Compute conditional mean
    cond_mean, interpolator = conditional_mean_1D_fft(Y, X, x_grid, h=h, kernel_type='gaussian')

    # Theoretical conditional mean: E[Y|X=x] = 2x + 1
    true_mean = 2 * x_grid + 1

    # Compare: mean absolute error should be small
    mae = np.mean(np.abs(cond_mean - true_mean))
    assert mae < 0.1, f"Mean absolute error too large: {mae}"

    # Test interpolator at a few points
    test_x = np.array([-1.5, 0.0, 1.5])
    interp_vals = interpolator(test_x)
    true_vals = 2 * test_x + 1
    assert np.allclose(interp_vals, true_vals, atol=0.1), "Interpolator values incorrect"

    print("Test passed: conditional_mean_1D_fft returns correct conditional mean.")

def test_conditional_mean_1D_fft_periodic():
    # Generate synthetic periodic data: Y = sin(2*pi*X) + noise, X in [0, 1]
    np.random.seed(42)
    N = 1000
    X = np.random.uniform(0, 1, N)
    Y = np.sin(2 * np.pi * X) + np.random.normal(0, 0.1, N)
    x_grid = np.linspace(0, 1, 200)
    h = 0.02

    # Compute conditional mean with periodic kernel
    cond_mean, interpolator = conditional_mean_1D_fft(Y, X, x_grid, h=h, kernel_type='wrapped')

    # Theoretical conditional mean: E[Y|X=x] = sin(2*pi*x)
    true_mean = np.sin(2 * np.pi * x_grid)

    # Compare: mean absolute error should be small
    mae = np.mean(np.abs(cond_mean - true_mean))
    assert mae < 0.1, f"Mean absolute error too large (periodic): {mae}"

    # Test interpolator at a few points
    test_x = np.array([0.1, 0.5, 0.9])
    interp_vals = interpolator(test_x)
    true_vals = np.sin(2 * np.pi * test_x)
    assert np.allclose(interp_vals, true_vals, atol=0.1), "Interpolator values incorrect (periodic)"

    print("Test passed: conditional_mean_1D_fft returns correct conditional mean with periodic kernel.")


# Test case for the derivative of the conditional mean 

def test_conditional_mean_1D_derivative_fft_periodic():
    np.random.seed(456)
    N = 20000
    X = np.random.uniform(0, 1, N)
    Y = np.sin(2 * np.pi * X)
    x_grid = np.linspace(0, 1, 1024)
    h = 0.0125

    dEdx, interpolator = conditional_mean_1D_derivative_fft(Y, X, x_grid, h=h, L=1.0)
    true_deriv = 2 * np.pi * np.cos(2 * np.pi * x_grid)

    mae = np.mean(np.abs(dEdx - true_deriv))
    print(f"MAE: {mae:.5f}")
    assert mae < 0.125, f"Mean absolute error too large: {mae}"

    test_x = np.array([0.1, 0.5, 0.9])
    interp_vals = interpolator(test_x)
    true_vals = 2 * np.pi * np.cos(2 * np.pi * test_x)
    assert np.allclose(interp_vals, true_vals, atol=0.25), "Interpolator incorrect"

    print("Test passed.")