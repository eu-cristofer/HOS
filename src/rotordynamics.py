"""
Rotordynamics Analysis Module

This module provides specialized tools for analyzing rotordynamic systems
and detecting common faults such as unbalance, misalignment, and bearing defects.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
from typing import Dict, List, Tuple, Optional
import warnings


class RotordynamicsAnalyzer:
    """
    A specialized analyzer for rotordynamic systems and fault detection.
    
    This class provides methods for analyzing vibration signals from rotating machinery
    and detecting common faults using spectral analysis techniques.
    """
    
    def __init__(self, fs: float = 1000.0):
        """
        Initialize the rotordynamics analyzer.
        
        Parameters:
        -----------
        fs : float
            Sampling frequency in Hz
        """
        self.fs = fs
        self.fault_features = {}
        
    def generate_rotor_signal(self, rpm: float, duration: float = 1.0, 
                            fault_type: str = 'healthy', noise_level: float = 0.1) -> np.ndarray:
        """
        Generate simulated rotor vibration signals with various fault conditions.
        
        Parameters:
        -----------
        rpm : float
            Rotor speed in revolutions per minute
        duration : float
            Signal duration in seconds
        fault_type : str
            Type of fault ('healthy', 'unbalance', 'misalignment', 'bearing_defect')
        noise_level : float
            Noise level as fraction of signal amplitude
            
        Returns:
        --------
        signal : np.ndarray
            Simulated vibration signal
        """
        t = np.linspace(0, duration, int(self.fs * duration), endpoint=False)
        freq_rpm = rpm / 60.0  # Convert RPM to Hz
        
        # Base signal (1X component)
        signal = np.sin(2 * np.pi * freq_rpm * t)
        
        if fault_type == 'healthy':
            # Only 1X component with small noise
            signal += noise_level * np.random.normal(0, 1, len(t))
            
        elif fault_type == 'unbalance':
            # Strong 1X component, small 2X
            signal = 2.0 * np.sin(2 * np.pi * freq_rpm * t)
            signal += 0.3 * np.sin(2 * np.pi * 2 * freq_rpm * t)
            signal += noise_level * np.random.normal(0, 1, len(t))
            
        elif fault_type == 'misalignment':
            # Strong 1X and 2X components
            signal = 1.5 * np.sin(2 * np.pi * freq_rpm * t)
            signal += 1.2 * np.sin(2 * np.pi * 2 * freq_rpm * t)
            signal += 0.2 * np.sin(2 * np.pi * 3 * freq_rpm * t)
            signal += noise_level * np.random.normal(0, 1, len(t))
            
        elif fault_type == 'bearing_defect':
            # Bearing defect frequencies (simplified model)
            # Ball pass frequency outer race (BPFO) ≈ 3.5 * rpm/60
            # Ball pass frequency inner race (BPFI) ≈ 4.5 * rpm/60
            bpfo = 3.5 * freq_rpm
            bpfi = 4.5 * freq_rpm
            
            signal = np.sin(2 * np.pi * freq_rpm * t)
            signal += 0.8 * np.sin(2 * np.pi * bpfo * t)
            signal += 0.6 * np.sin(2 * np.pi * bpfi * t)
            signal += 0.3 * np.sin(2 * np.pi * 2 * bpfo * t)
            signal += noise_level * np.random.normal(0, 1, len(t))
            
        else:
            raise ValueError(f"Unknown fault type: {fault_type}")
            
        return signal
    
    def extract_fault_features(self, signal: np.ndarray, rpm: float) -> Dict[str, float]:
        """
        Extract features that are characteristic of different fault types.
        
        Parameters:
        -----------
        signal : np.ndarray
            Vibration signal
        rpm : float
            Rotor speed in RPM
            
        Returns:
        --------
        features : dict
            Dictionary of extracted features
        """
        freq_rpm = rpm / 60.0
        
        # Compute FFT
        freqs = fftfreq(len(signal), 1/self.fs)
        X = fft(signal)
        magnitude = np.abs(X)
        
        # Find peaks in the spectrum
        peaks, properties = signal.find_peaks(magnitude, height=np.max(magnitude) * 0.1)
        peak_freqs = freqs[peaks]
        peak_mags = magnitude[peaks]
        
        # Extract features
        features = {}
        
        # 1X, 2X, 3X components
        for harmonic in [1, 2, 3]:
            target_freq = harmonic * freq_rpm
            # Find closest peak to target frequency
            idx = np.argmin(np.abs(peak_freqs - target_freq))
            if np.abs(peak_freqs[idx] - target_freq) < freq_rpm * 0.1:  # Within 10% tolerance
                features[f'{harmonic}X_magnitude'] = peak_mags[idx]
            else:
                features[f'{harmonic}X_magnitude'] = 0.0
        
        # RMS value
        features['rms'] = np.sqrt(np.mean(signal**2))
        
        # Peak-to-peak value
        features['peak_to_peak'] = np.max(signal) - np.min(signal)
        
        # Crest factor (peak-to-peak / RMS)
        features['crest_factor'] = features['peak_to_peak'] / features['rms'] if features['rms'] > 0 else 0
        
        # Kurtosis (measure of impulsiveness)
        features['kurtosis'] = np.mean((signal - np.mean(signal))**4) / (np.std(signal)**4)
        
        # Skewness
        features['skewness'] = np.mean((signal - np.mean(signal))**3) / (np.std(signal)**3)
        
        # Spectral centroid
        features['spectral_centroid'] = np.sum(freqs[:len(freqs)//2] * magnitude[:len(magnitude)//2]) / np.sum(magnitude[:len(magnitude)//2])
        
        # Total harmonic distortion (THD)
        fundamental_idx = np.argmin(np.abs(freqs - freq_rpm))
        fundamental_mag = magnitude[fundamental_idx]
        harmonic_power = 0
        for harmonic in range(2, 6):  # 2X to 5X
            harmonic_freq = harmonic * freq_rpm
            harmonic_idx = np.argmin(np.abs(freqs - harmonic_freq))
            if np.abs(freqs[harmonic_idx] - harmonic_freq) < freq_rpm * 0.1:
                harmonic_power += magnitude[harmonic_idx]**2
        
        features['thd'] = np.sqrt(harmonic_power) / fundamental_mag if fundamental_mag > 0 else 0
        
        return features
    
    def plot_orbit(self, x_signal: np.ndarray, y_signal: np.ndarray, 
                   title: str = "Rotor Orbit") -> None:
        """
        Plot rotor orbit from X and Y vibration signals.
        
        Parameters:
        -----------
        x_signal : np.ndarray
            X-direction vibration signal
        y_signal : np.ndarray
            Y-direction vibration signal
        title : str
            Plot title
        """
        plt.figure(figsize=(8, 8))
        plt.plot(x_signal, y_signal, 'b-', alpha=0.7)
        plt.plot(x_signal[0], y_signal[0], 'go', markersize=8, label='Start')
        plt.plot(x_signal[-1], y_signal[-1], 'ro', markersize=8, label='End')
        plt.xlabel('X Displacement')
        plt.ylabel('Y Displacement')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def plot_spectrum_with_harmonics(self, signal: np.ndarray, rpm: float,
                                   title: str = "Spectrum with Harmonics") -> None:
        """
        Plot spectrum with harmonic markers.
        
        Parameters:
        -----------
        signal : np.ndarray
            Vibration signal
        rpm : float
            Rotor speed in RPM
        title : str
            Plot title
        """
        freq_rpm = rpm / 60.0
        
        # Compute FFT
        freqs = fftfreq(len(signal), 1/self.fs)
        X = fft(signal)
        magnitude = np.abs(X)
        
        # Plot only positive frequencies
        pos_mask = freqs >= 0
        freqs_pos = freqs[pos_mask]
        magnitude_pos = magnitude[pos_mask]
        
        plt.figure(figsize=(12, 6))
        plt.semilogy(freqs_pos, magnitude_pos, 'b-', alpha=0.7)
        
        # Mark harmonics
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        for harmonic in range(1, 6):
            harmonic_freq = harmonic * freq_rpm
            if harmonic_freq < freqs_pos[-1]:
                plt.axvline(harmonic_freq, color=colors[harmonic-1], 
                           linestyle='--', alpha=0.8, 
                           label=f'{harmonic}X ({harmonic_freq:.1f} Hz)')
        
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def classify_fault(self, features: Dict[str, float]) -> str:
        """
        Simple fault classification based on extracted features.
        
        Parameters:
        -----------
        features : dict
            Extracted features
            
        Returns:
        --------
        fault_type : str
            Predicted fault type
        """
        # Simple rule-based classification
        if features['2X_magnitude'] > features['1X_magnitude'] * 0.8:
            return 'misalignment'
        elif features['1X_magnitude'] > 2.0 and features['2X_magnitude'] < features['1X_magnitude'] * 0.5:
            return 'unbalance'
        elif features['kurtosis'] > 4.0:
            return 'bearing_defect'
        else:
            return 'healthy'


def generate_rotor_dataset(rpm_range: Tuple[float, float] = (1000, 3000),
                          fault_types: List[str] = None,
                          n_samples_per_fault: int = 50) -> Dict:
    """
    Generate a dataset of rotor signals for machine learning.
    
    Parameters:
    -----------
    rpm_range : tuple
        Range of RPM values (min, max)
    fault_types : list
        List of fault types to include
    n_samples_per_fault : int
        Number of samples per fault type
        
    Returns:
    --------
    dataset : dict
        Dictionary containing signals, labels, and features
    """
    if fault_types is None:
        fault_types = ['healthy', 'unbalance', 'misalignment', 'bearing_defect']
    
    analyzer = RotordynamicsAnalyzer()
    
    signals = []
    labels = []
    features_list = []
    
    for fault_type in fault_types:
        for i in range(n_samples_per_fault):
            # Random RPM in range
            rpm = np.random.uniform(rpm_range[0], rpm_range[1])
            
            # Generate signal
            signal = analyzer.generate_rotor_signal(rpm, duration=1.0, 
                                                  fault_type=fault_type)
            
            # Extract features
            features = analyzer.extract_fault_features(signal, rpm)
            
            signals.append(signal)
            labels.append(fault_type)
            features_list.append(features)
    
    return {
        'signals': np.array(signals),
        'labels': labels,
        'features': features_list,
        'feature_names': list(features_list[0].keys()) if features_list else []
    }


if __name__ == "__main__":
    # Example usage
    analyzer = RotordynamicsAnalyzer(fs=1000.0)
    
    # Generate signals for different fault types
    rpm = 1800
    fault_types = ['healthy', 'unbalance', 'misalignment', 'bearing_defect']
    
    for fault_type in fault_types:
        signal = analyzer.generate_rotor_signal(rpm, fault_type=fault_type)
        
        # Extract features
        features = analyzer.extract_fault_features(signal, rpm)
        
        # Plot spectrum
        analyzer.plot_spectrum_with_harmonics(signal, rpm, 
                                            f"Spectrum - {fault_type.title()}")
        
        # Classify fault
        predicted = analyzer.classify_fault(features)
        print(f"True: {fault_type}, Predicted: {predicted}")
        print(f"Features: {features}")
        print("-" * 50)

