from sys import stdout
import numpy as np

def mask_diagonal(masked_array):
    """ Given a masked array, it returns the same array with the diagonals masked"""
    if len(masked_array.shape) == 3:
        i, j, k = np.meshgrid(
            *map(np.arange, masked_array.shape), indexing='ij')
        masked_array.mask = (i == j)
    elif len(masked_array.shape) == 2:
        i, j = np.meshgrid(
            *map(np.arange, masked_array.shape), indexing='ij')
        masked_array.mask = (i == j)

    return masked_array
