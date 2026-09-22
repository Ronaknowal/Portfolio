"""Bounded FFT blocks, a reusable filter spectrum, and the ordinary SciPy route.

NumPy 2.3.5 and SciPy 1.18.1 targets. Real, finite, nonempty 1-D arrays.
The output is the first len(values) samples of linear causal convolution.
"""
import numpy as np
from scipy.signal import oaconvolve
from convolution_mechanisms import direct, overlap_add


def overlap_add_fft(values, kernel, block_size):
    values, kernel = np.asarray(values, dtype=float), np.asarray(kernel, dtype=float)
    if values.ndim != 1 or kernel.ndim != 1 or min(len(values), len(kernel)) == 0:
        raise ValueError("Need two nonempty vectors")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    size = 1 << (block_size + len(kernel)-2).bit_length()
    filter_spectrum = np.fft.rfft(kernel, n=size)
    # All requested output samples must be retained; no all-pairs Toeplitz matrix.
    output = np.zeros(len(values))
    for start in range(0, len(values), block_size):
        block = values[start:start+block_size]
        transformed = np.fft.rfft(block, n=size)
        convolved = np.fft.irfft(transformed*filter_spectrum, n=size)
        count = min(len(block)+len(kernel)-1, len(values)-start)
        output[start:start+count] += convolved[:count]
    return output


def main():
    values = np.array([2., -1., 3., 0., 1., 2., -.5])
    kernel = np.array([.5, 1., -.25, .125])
    expected = direct(values, kernel)
    library = oaconvolve(values, kernel, mode="full")[:len(values)]
    np.testing.assert_allclose(library, expected, atol=1e-12)
    for block_size in (1, 2, 3, 8):
        actual = overlap_add_fft(values, kernel, block_size)
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        np.testing.assert_allclose(actual, overlap_add(values, kernel, block_size), atol=1e-12)
        print("block", block_size, "output", actual, "maximum error", abs(actual-expected).max())


if __name__ == "__main__":
    main()
