"""
Spectral Analysis Module for Higher Order Spectra

This module provides implementations of various spectral analysis techniques
including FFT, bispectrum, trispectrum, and related higher-order spectral methods.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq, fftshift
from typing import Tuple, Optional, Union
import warnings


class SpectralAnalyzer:
    """
    A comprehensive spectral analysis class for signal processing applications.
    
    This class provides methods for computing various spectral representations
    including power spectral density, bispectrum, and trispectrum.
    """
    
    def __init__(self, fs: float = 1.0):
        """
        Initialize the spectral analyzer.
        
        Parameters:
        -----------
        fs : float
            Sampling frequency in Hz
        """
        self.fs = fs
        
    def compute_fft(self, x: np.ndarray, window: str = 'hann') -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the FFT of a signal with optional windowing.
        
        Parameters:
        -----------
        x : np.ndarray
            Input signal
        window : str
            Window type ('hann', 'hamming', 'blackman', 'rectangular')
            
        Returns:
        --------
        freqs : np.ndarray
            Frequency array
        X : np.ndarray
            Complex FFT coefficients
        """
        N = len(x)
        
        # Apply window
        if window == 'rectangular':
            w = np.ones(N)
        elif window == 'hann':
            w = signal.windows.hann(N)
        elif window == 'hamming':
            w = signal.windows.hamming(N)
        elif window == 'blackman':
            w = signal.windows.blackman(N)
        else:
            raise ValueError(f"Unknown window type: {window}")
            
        # Apply window and compute FFT
        x_windowed = x * w
        X = fft(x_windowed)
        freqs = fftfreq(N, 1/self.fs)
        
        return freqs, X
    
    def compute_psd(self, x: np.ndarray, nperseg: Optional[int] = None, 
                   window: str = 'hann', overlap: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Power Spectral Density using Welch's method.
        
        Parameters:
        -----------
        x : np.ndarray
            Input signal
        nperseg : int, optional
            Length of each segment
        window : str
            Window type
        overlap : float
            Overlap ratio between segments
            
        Returns:
        --------
        freqs : np.ndarray
            Frequency array
        psd : np.ndarray
            Power spectral density
        """
        if nperseg is None:
            nperseg = min(len(x) // 4, 1024)
            
        freqs, psd = signal.welch(x, fs=self.fs, nperseg=nperseg, 
                                 window=window, noverlap=int(nperseg * overlap))
        return freqs, psd
    
    def compute_bispectrum(self, x: np.ndarray, nfft: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the bispectrum of a signal.
        
        The bispectrum is the Fourier transform of the third-order cumulant.
        
        Parameters:
        -----------
        x : np.ndarray
            Input signal
        nfft : int, optional
            FFT length
            
        Returns:
        --------
        f1 : np.ndarray
            First frequency axis
        f2 : np.ndarray
            Second frequency axis
        bispec : np.ndarray
            Bispectrum magnitude
        """
        if nfft is None:
            nfft = 2 ** int(np.ceil(np.log2(len(x))))
            
        # Compute third-order cumulant
        N = len(x)
        max_lag = min(N // 4, 64)
        
        # Estimate third-order cumulant
        cumulant = np.zeros((2*max_lag+1, 2*max_lag+1), dtype=complex)
        
        for tau1 in range(-max_lag, max_lag+1):
            for tau2 in range(-max_lag, max_lag+1):
                if tau1 >= 0 and tau2 >= 0:
                    # Compute E[x(n)x(n+tau1)x(n+tau2)]
                    valid_range = slice(0, N - max(tau1, tau2))
                    cumulant[tau1+max_lag, tau2+max_lag] = np.mean(
                        x[valid_range] * x[valid_range + tau1] * x[valid_range + tau2]
                    )
        
        # Compute bispectrum via 2D FFT
        bispec = np.fft.fft2(cumulant, s=(nfft, nfft))
        
        # Create frequency axes
        f1 = fftfreq(nfft, 1/self.fs)
        f2 = fftfreq(nfft, 1/self.fs)
        
        return f1, f2, np.abs(bispec)
    
    def compute_trispectrum(self, x: np.ndarray, nfft: Optional[int] = None) -> np.ndarray:
        """
        Compute the trispectrum of a signal.
        
        The trispectrum is the Fourier transform of the fourth-order cumulant.
        
        Parameters:
        -----------
        x : np.ndarray
            Input signal
        nfft : int, optional
            FFT length
            
        Returns:
        --------
        trispec : np.ndarray
            Trispectrum magnitude (4D array)
        """
        if nfft is None:
            nfft = 2 ** int(np.ceil(np.log2(len(x))))
            
        N = len(x)
        max_lag = min(N // 8, 32)  # Smaller for computational efficiency
        
        # Estimate fourth-order cumulant
        cumulant = np.zeros((2*max_lag+1, 2*max_lag+1, 2*max_lag+1), dtype=complex)
        
        for tau1 in range(-max_lag, max_lag+1):
            for tau2 in range(-max_lag, max_lag+1):
                for tau3 in range(-max_lag, max_lag+1):
                    if all(tau >= 0 for tau in [tau1, tau2, tau3]):
                        # Compute E[x(n)x(n+tau1)x(n+tau2)x(n+tau3)]
                        max_tau = max(tau1, tau2, tau3)
                        valid_range = slice(0, N - max_tau)
                        cumulant[tau1+max_lag, tau2+max_lag, tau3+max_lag] = np.mean(
                            x[valid_range] * x[valid_range + tau1] * 
                            x[valid_range + tau2] * x[valid_range + tau3]
                        )
        
        # Compute trispectrum via 3D FFT
        trispec = np.fft.fftn(cumulant, s=(nfft, nfft, nfft))
        
        return np.abs(trispec)
    
    def plot_spectrum(self, freqs: np.ndarray, spectrum: np.ndarray, 
                     title: str = "Spectrum", log_scale: bool = True) -> None:
        """
        Plot spectrum with proper formatting.
        
        Parameters:
        -----------
        freqs : np.ndarray
            Frequency array
        spectrum : np.ndarray
            Spectrum magnitude
        title : str
            Plot title
        log_scale : bool
            Whether to use logarithmic scale
        """
        plt.figure(figsize=(10, 6))
        
        if log_scale:
            plt.semilogy(freqs, spectrum)
        else:
            plt.plot(freqs, spectrum)
            
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_bispectrum(self, f1: np.ndarray, f2: np.ndarray, bispec: np.ndarray,
                       title: str = "Bispectrum") -> None:
        """
        Plot bispectrum as a 2D heatmap.
        
        Parameters:
        -----------
        f1 : np.ndarray
            First frequency axis
        f2 : np.ndarray
            Second frequency axis
        bispec : np.ndarray
            Bispectrum magnitude
        title : str
            Plot title
        """
        plt.figure(figsize=(10, 8))
        
        # Plot only positive frequencies
        pos_mask = (f1 >= 0) & (f2 >= 0)
        f1_pos = f1[pos_mask]
        f2_pos = f2[pos_mask]
        bispec_pos = bispec[pos_mask]
        
        # Create 2D grid for plotting
        f1_grid, f2_grid = np.meshgrid(f1_pos, f2_pos)
        bispec_grid = np.zeros_like(f1_grid)
        
        # Fill the grid (this is a simplified approach)
        for i, freq1 in enumerate(f1_pos):
            for j, freq2 in enumerate(f2_pos):
                idx1 = np.argmin(np.abs(f1 - freq1))
                idx2 = np.argmin(np.abs(f2 - freq2))
                bispec_grid[j, i] = bispec[idx1, idx2]
        
        plt.contourf(f1_grid, f2_grid, bispec_grid, levels=50, cmap='viridis')
        plt.colorbar(label='Bispectrum Magnitude')
        plt.xlabel('Frequency f1 (Hz)')
        plt.ylabel('Frequency f2 (Hz)')
        plt.title(title)
        plt.tight_layout()
        plt.show()


def generate_test_signals(fs: float = 1000.0, duration: float = 1.0) -> dict:
    """
    Generate test signals for spectral analysis.
    
    Parameters:
    -----------
    fs : float
        Sampling frequency
    duration : float
        Signal duration in seconds
        
    Returns:
    --------
    signals : dict
        Dictionary containing different test signals
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    
    # Pure sinusoid
    f1 = 50.0
    sinusoid = np.sin(2 * np.pi * f1 * t)
    
    # Multi-frequency signal
    f2 = 120.0
    multifreq = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)
    
    # Chirp signal
    chirp = signal.chirp(t, f0=10, f1=200, t1=duration, method='linear')
    
    # Signal with harmonics
    harmonics = (np.sin(2 * np.pi * f1 * t) + 
                0.3 * np.sin(2 * np.pi * 2 * f1 * t) + 
                0.1 * np.sin(2 * np.pi * 3 * f1 * t))
    
    # Noisy signal
    noise = np.random.normal(0, 0.1, len(t))
    noisy_signal = multifreq + noise
    
    return {
        'time': t,
        'sinusoid': sinusoid,
        'multifreq': multifreq,
        'chirp': chirp,
        'harmonics': harmonics,
        'noisy': noisy_signal
    }


if __name__ == "__main__":
    # Example usage
    fs = 1000.0
    analyzer = SpectralAnalyzer(fs)
    
    # Generate test signals
    signals = generate_test_signals(fs)
    
    # Analyze multifrequency signal
    freqs, X = analyzer.compute_fft(signals['multifreq'])
    analyzer.plot_spectrum(freqs, np.abs(X), "FFT of Multi-frequency Signal")
    
    # Compute PSD
    freqs_psd, psd = analyzer.compute_psd(signals['multifreq'])
    analyzer.plot_spectrum(freqs_psd, psd, "Power Spectral Density", log_scale=True)

