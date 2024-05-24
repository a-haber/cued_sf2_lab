from cued_sf2_lab.laplacian_pyramid import bpp, quantise
from cued_sf2_lab.dct import dct_ii, dct_iv, colxfm, regroup
from cued_sf2_lab.familiarisation import load_mat_img, plot_image
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

def dctbpp(Yr, N):
    """Calculate the total number of bits from a re-grouped image Yr.

    Uses bpp(Ys) on each sub-image Ys of Yr, then multiplies each result
    by the number of pixels in the sub-image, summing to give the total number of bits.
    """
    w = int(Yr.shape[0] / N)  # width of each sub-image
    total_bits = 0
    for i in range(N):
        for j in range(N):
            Ys = Yr[i*w:(i+1)*w, j*w:(j+1)*w]
            total_bits += bpp(Ys) * Ys.size
    return total_bits


def dct_n_by_n(image, N):
    """Compute the DCT transform of an image using an N x N DCT block."""
    C = dct_ii(N)
    return colxfm(colxfm(image, C).T, C).T


def idct_n_by_n(image, N):
    """Compute the inverse DCT transform of an image using an N x N DCT block."""
    C = dct_ii(N)
    return colxfm(colxfm(image.T, C.T).T, C.T)


def calculate_dct_rms(image, N, quantisation_function, *quant_args):
    """Calculate rms error from NxN DCT transform and quantisation.
    
    Parameters:
    - image: the input image to be transformed
    - N: size of the DCT (N x N)
    - quantisation_function: the function with which the transformed image is quantised
    - *quant_args: additional arguments for the quantisation function as necessary"""
    dct_img = dct_n_by_n(image, N)
    quantised_img = quantisation_function(dct_img, *quant_args)
    reconstructed_img = idct_n_by_n(quantised_img, N)
    rms = np.std(image - reconstructed_img)
    return rms


def objective_function_dct(image, N, quantisation_function, *quant_args):
    reference_error = np.std(image - quantise(image, 17))
    reconstruction_error = calculate_dct_rms(image, N, quantisation_function, *quant_args)
    relative_error = np.abs(reference_error - reconstruction_error)
    return relative_error


# load image and define relevant parameters
lighthouse, _ = load_mat_img(img='lighthouse.mat', img_info='X')
lighthouse = lighthouse - 128.0
N = 8

for rise_1 in [0.5, 1.0, 1.5]:


    # Optimise step size to match reference rms
    result = minimize_scalar(lambda step_size: objective_function_dct(lighthouse, N, quantise, step_size, rise_1*step_size), bounds=(0, 30), method='bounded')

    # Extract the optimal step size and the corresponding minimum error
    required_step_size = result.x
    min_error = result.fun

    Y = dct_n_by_n(lighthouse, N)
    Yq = quantise(Y, required_step_size, rise_1*required_step_size)
    Yr_DCT = regroup(Yq, N) / N
    Z = idct_n_by_n(Yq, N)

    # Print the results
    print(f"With rise1 set to {rise_1}:")
    print(required_step_size, "required step size")
    print(min_error, "min error relative to reference rms")
    print(np.std(lighthouse - Z), "objective rms error")
    print(dctbpp(Yr_DCT, N), "bits estimated")
