"""
Validation Framework for Spectral Analysis Methods

This module provides tools for validating and comparing different spectral analysis
methods, ensuring accuracy and reliability of the implemented algorithms.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
from typing import Dict, List, Tuple, Any, Optional
import warnings
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Container for validation results."""
    method_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    computation_time: float
    memory_usage: float
    parameters: Dict[str, Any]


class SpectralValidation:
    """
    Validation framework for spectral analysis methods.
    
    This class provides methods for validating spectral analysis algorithms
    against known theoretical results and comparing different approaches.
    """
    
    def __init__(self, fs: float = 1000.0):
        """
        Initialize the validation framework.
        
        Parameters:
        -----------
        fs : float
            Sampling frequency in Hz
        """
        self.fs = fs
        self.results = []
    
    def validate_fft_accuracy(self, n_trials: int = 100) -> ValidationResult:
        """
        Validate FFT accuracy against theoretical results.
        
        Parameters:
        -----------
        n_trials : int
            Number of validation trials
            
        Returns:
        --------
        result : ValidationResult
            Validation results
        """
        import time
        
        start_time = time.time()
        
        # Test parameters
        frequencies = [10, 25, 50, 100, 200]  # Hz
        amplitudes = [1.0, 2.0, 0.5, 1.5]
        durations = [0.5, 1.0, 2.0]  # seconds
        
        errors = []
        
        for trial in range(n_trials):
            # Random test parameters
            f0 = np.random.choice(frequencies)
            A = np.random.choice(amplitudes)
            duration = np.random.choice(durations)
            
            # Generate test signal
            t = np.linspace(0, duration, int(self.fs * duration), endpoint=False)
            x = A * np.sin(2 * np.pi * f0 * t)
            
            # Compute FFT
            X = fft(x)
            freqs = fftfreq(len(x), 1/self.fs)
            
            # Find peak frequency
            pos_mask = freqs >= 0
            magnitude = np.abs(X[pos_mask])
            peak_idx = np.argmax(magnitude)
            detected_freq = freqs[pos_mask][peak_idx]
            
            # Calculate error
            freq_error = abs(detected_freq - f0) / f0
            errors.append(freq_error)
        
        # Calculate metrics
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        max_error = np.max(errors)
        
        # Accuracy (1 - relative error)
        accuracy = 1 - mean_error
        
        computation_time = time.time() - start_time
        
        result = ValidationResult(
            method_name="FFT Accuracy",
            accuracy=accuracy,
            precision=1 - std_error,
            recall=1 - max_error,
            f1_score=2 * accuracy * (1 - std_error) / (accuracy + (1 - std_error)),
            computation_time=computation_time,
            memory_usage=0,  # Placeholder
            parameters={
                "n_trials": n_trials,
                "mean_error": mean_error,
                "std_error": std_error,
                "max_error": max_error
            }
        )
        
        self.results.append(result)
        return result
    
    def validate_psd_estimation(self, n_trials: int = 50) -> ValidationResult:
        """
        Validate PSD estimation accuracy.
        
        Parameters:
        -----------
        n_trials : int
            Number of validation trials
            
        Returns:
        --------
        result : ValidationResult
            Validation results
        """
        import time
        
        start_time = time.time()
        
        # Test parameters
        frequencies = [25, 50, 75, 100]
        amplitudes = [1.0, 2.0]
        noise_levels = [0.1, 0.2, 0.5]
        
        errors = []
        
        for trial in range(n_trials):
            # Random test parameters
            f0 = np.random.choice(frequencies)
            A = np.random.choice(amplitudes)
            noise_level = np.random.choice(noise_levels)
            
            # Generate test signal
            duration = 2.0
            t = np.linspace(0, duration, int(self.fs * duration), endpoint=False)
            x = A * np.sin(2 * np.pi * f0 * t) + noise_level * np.random.normal(0, 1, len(t))
            
            # Compute PSD using Welch's method
            freqs_psd, psd = signal.welch(x, fs=self.fs, nperseg=256, window='hann')
            
            # Find peak frequency
            peak_idx = np.argmax(psd)
            detected_freq = freqs_psd[peak_idx]
            
            # Calculate error
            freq_error = abs(detected_freq - f0) / f0
            errors.append(freq_error)
        
        # Calculate metrics
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        accuracy = 1 - mean_error
        
        computation_time = time.time() - start_time
        
        result = ValidationResult(
            method_name="PSD Estimation",
            accuracy=accuracy,
            precision=1 - std_error,
            recall=accuracy,
            f1_score=accuracy,
            computation_time=computation_time,
            memory_usage=0,
            parameters={
                "n_trials": n_trials,
                "mean_error": mean_error,
                "std_error": std_error
            }
        )
        
        self.results.append(result)
        return result
    
    def validate_bispectrum_properties(self) -> ValidationResult:
        """
        Validate bispectrum mathematical properties.
        
        Returns:
        --------
        result : ValidationResult
            Validation results
        """
        import time
        
        start_time = time.time()
        
        # Generate test signal
        duration = 1.0
        t = np.linspace(0, duration, int(self.fs * duration), endpoint=False)
        f1, f2 = 25.0, 50.0
        x = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)
        
        # Compute bispectrum (simplified)
        nfft = 128
        N = len(x)
        max_lag = min(N // 4, 32)
        
        # Estimate third-order cumulant
        cumulant = np.zeros((2*max_lag+1, 2*max_lag+1), dtype=complex)
        
        for tau1 in range(-max_lag, max_lag+1):
            for tau2 in range(-max_lag, max_lag+1):
                if tau1 >= 0 and tau2 >= 0:
                    valid_range = slice(0, N - max(tau1, tau2))
                    cumulant[tau1+max_lag, tau2+max_lag] = np.mean(
                        x[valid_range] * x[valid_range + tau1] * x[valid_range + tau2]
                    )
        
        # Compute bispectrum
        bispec = np.fft.fft2(cumulant, s=(nfft, nfft))
        
        # Check symmetry properties
        bispec_mag = np.abs(bispec)
        
        # Check if bispectrum is symmetric (simplified check)
        symmetry_error = 0
        for i in range(nfft//2):
            for j in range(nfft//2):
                if i != j:
                    symmetry_error += abs(bispec_mag[i, j] - bispec_mag[j, i])
        
        symmetry_error /= (nfft//2)**2
        
        computation_time = time.time() - start_time
        
        result = ValidationResult(
            method_name="Bispectrum Properties",
            accuracy=1 - symmetry_error,
            precision=1 - symmetry_error,
            recall=1 - symmetry_error,
            f1_score=1 - symmetry_error,
            computation_time=computation_time,
            memory_usage=0,
            parameters={
                "symmetry_error": symmetry_error,
                "nfft": nfft,
                "max_lag": max_lag
            }
        )
        
        self.results.append(result)
        return result
    
    def compare_window_functions(self, signal_type: str = 'multifreq') -> Dict[str, ValidationResult]:
        """
        Compare different window functions.
        
        Parameters:
        -----------
        signal_type : str
            Type of test signal
            
        Returns:
        --------
        results : dict
            Comparison results for different windows
        """
        import time
        
        # Generate test signal
        duration = 1.0
        t = np.linspace(0, duration, int(self.fs * duration), endpoint=False)
        
        if signal_type == 'multifreq':
            f1, f2 = 25.0, 75.0
            x = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)
        else:
            f0 = 50.0
            x = np.sin(2 * np.pi * f0 * t)
        
        windows = ['rectangular', 'hann', 'hamming', 'blackman']
        results = {}
        
        for window_name in windows:
            start_time = time.time()
            
            # Apply window
            if window_name == 'rectangular':
                w = np.ones(len(x))
            elif window_name == 'hann':
                w = signal.windows.hann(len(x))
            elif window_name == 'hamming':
                w = signal.windows.hamming(len(x))
            elif window_name == 'blackman':
                w = signal.windows.blackman(len(x))
            
            x_windowed = x * w
            X = fft(x_windowed)
            freqs = fftfreq(len(x), 1/self.fs)
            
            # Calculate main lobe width and sidelobe level
            magnitude = np.abs(X)
            pos_mask = freqs >= 0
            magnitude_pos = magnitude[pos_mask]
            freqs_pos = freqs[pos_mask]
            
            # Find main lobe width (simplified)
            peak_idx = np.argmax(magnitude_pos)
            peak_freq = freqs_pos[peak_idx]
            
            # Find -3dB points
            peak_mag = magnitude_pos[peak_idx]
            threshold = peak_mag / np.sqrt(2)
            
            left_idx = peak_idx
            right_idx = peak_idx
            
            while left_idx > 0 and magnitude_pos[left_idx] > threshold:
                left_idx -= 1
            
            while right_idx < len(magnitude_pos) - 1 and magnitude_pos[right_idx] > threshold:
                right_idx += 1
            
            main_lobe_width = freqs_pos[right_idx] - freqs_pos[left_idx]
            
            # Calculate sidelobe level (simplified)
            sidelobe_level = np.max(magnitude_pos) / np.max(magnitude_pos[peak_idx//2:peak_idx*2])
            
            computation_time = time.time() - start_time
            
            result = ValidationResult(
                method_name=f"Window: {window_name}",
                accuracy=1 / main_lobe_width,  # Higher accuracy for narrower main lobe
                precision=1 / sidelobe_level,  # Higher precision for lower sidelobes
                recall=1 / main_lobe_width,
                f1_score=2 / (main_lobe_width + sidelobe_level),
                computation_time=computation_time,
                memory_usage=0,
                parameters={
                    "main_lobe_width": main_lobe_width,
                    "sidelobe_level": sidelobe_level,
                    "peak_frequency": peak_freq
                }
            )
            
            results[window_name] = result
            self.results.append(result)
        
        return results
    
    def plot_validation_results(self) -> None:
        """
        Plot validation results.
        """
        if not self.results:
            print("No validation results to plot.")
            return
        
        # Extract data
        methods = [r.method_name for r in self.results]
        accuracies = [r.accuracy for r in self.results]
        precisions = [r.precision for r in self.results]
        f1_scores = [r.f1_score for r in self.results]
        times = [r.computation_time for r in self.results]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy comparison
        axes[0, 0].bar(methods, accuracies, alpha=0.7)
        axes[0, 0].set_title('Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Precision comparison
        axes[0, 1].bar(methods, precisions, alpha=0.7, color='orange')
        axes[0, 1].set_title('Precision Comparison')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # F1 Score comparison
        axes[1, 0].bar(methods, f1_scores, alpha=0.7, color='green')
        axes[1, 0].set_title('F1 Score Comparison')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Computation time comparison
        axes[1, 1].bar(methods, times, alpha=0.7, color='red')
        axes[1, 1].set_title('Computation Time Comparison')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def generate_validation_report(self) -> str:
        """
        Generate a comprehensive validation report.
        
        Returns:
        --------
        report : str
            Validation report
        """
        if not self.results:
            return "No validation results available."
        
        report = "SPECTRAL ANALYSIS VALIDATION REPORT\n"
        report += "=" * 50 + "\n\n"
        
        for result in self.results:
            report += f"Method: {result.method_name}\n"
            report += f"  Accuracy: {result.accuracy:.4f}\n"
            report += f"  Precision: {result.precision:.4f}\n"
            report += f"  Recall: {result.recall:.4f}\n"
            report += f"  F1 Score: {result.f1_score:.4f}\n"
            report += f"  Computation Time: {result.computation_time:.4f} seconds\n"
            report += f"  Parameters: {result.parameters}\n\n"
        
        # Summary statistics
        avg_accuracy = np.mean([r.accuracy for r in self.results])
        avg_precision = np.mean([r.precision for r in self.results])
        avg_f1 = np.mean([r.f1_score for r in self.results])
        total_time = sum([r.computation_time for r in self.results])
        
        report += "SUMMARY STATISTICS\n"
        report += "-" * 20 + "\n"
        report += f"Average Accuracy: {avg_accuracy:.4f}\n"
        report += f"Average Precision: {avg_precision:.4f}\n"
        report += f"Average F1 Score: {avg_f1:.4f}\n"
        report += f"Total Computation Time: {total_time:.4f} seconds\n"
        
        return report


def run_comprehensive_validation() -> SpectralValidation:
    """
    Run comprehensive validation of all spectral analysis methods.
    
    Returns:
    --------
    validator : SpectralValidation
        Validation results
    """
    print("Running comprehensive validation...")
    
    validator = SpectralValidation()
    
    # Run all validation tests
    print("1. Validating FFT accuracy...")
    validator.validate_fft_accuracy()
    
    print("2. Validating PSD estimation...")
    validator.validate_psd_estimation()
    
    print("3. Validating bispectrum properties...")
    validator.validate_bispectrum_properties()
    
    print("4. Comparing window functions...")
    validator.compare_window_functions()
    
    print("5. Generating validation report...")
    report = validator.generate_validation_report()
    print(report)
    
    print("6. Plotting results...")
    validator.plot_validation_results()
    
    return validator


if __name__ == "__main__":
    # Run validation
    validator = run_comprehensive_validation()
    
    # Save results
    import json
    results_data = []
    for result in validator.results:
        results_data.append({
            'method_name': result.method_name,
            'accuracy': result.accuracy,
            'precision': result.precision,
            'recall': result.recall,
            'f1_score': result.f1_score,
            'computation_time': result.computation_time,
            'parameters': result.parameters
        })
    
    with open('validation_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print("Validation complete! Results saved to validation_results.json")

