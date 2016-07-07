from sys import stdout
import numpy as np

class ProgressBar():
    """" Custom Progress Bar used during classification process """
    def __init__(self, total, start=False):
        """
        total - total number of events
        start - automatically start when created? 
        """
        self.total = total
        self.current = 0.0
        self.last_int = 0
        if start:
            self.next()

    def update_progress(self, progress):
        display = '\r[{0}] {1}%'.format('#' * (progress / 10), progress)
        stdout.write(display)
        stdout.flush()

    def next(self):
        """ Increment progress bar"""
        if not self.last_int == int((self.current) / self.total * 100):
            self.update_progress(int((self.current) / self.total * 100))
            self.last_int = int((self.current) / self.total * 100)

        if self.current == self.total:
            self.reset()
        else:
            self.current = self.current + 1

    def reset(self):
        print ""
        self.current = 0.0

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