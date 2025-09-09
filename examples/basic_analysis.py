#!/usr/bin/env python3
"""
Basic Spectral Analysis Examples

This script demonstrates basic spectral analysis techniques using the HOS package.
It serves as a starting point for learning and experimentation.

Author: Cristofer Antoni Souza Costa
Institution: Federal University of Uberlândia
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.append(os.path.join('..', 'src'))

from spectral_analysis import SpectralAnalyzer, generate_test_signals
from rotordynamics import RotordynamicsAnalyzer, generate_rotor_dataset
from ml_classifier import FaultClassifier, compare_models


def example_1_basic_fft():
    """Example 1: Basic FFT analysis of a sinusoidal signal."""
    print("Example 1: Basic FFT Analysis")
    print("=" * 40)
    
    # Signal parameters
    fs = 1000.0  # Sampling frequency
    duration = 1.0  # Signal duration
    f0 = 50.0  # Signal frequency
    A = 2.0  # Signal amplitude
    
    # Generate signal
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    x = A * np.sin(2 * np.pi * f0 * t)
    
    # Analyze with SpectralAnalyzer
    analyzer = SpectralAnalyzer(fs)
    freqs, X = analyzer.compute_fft(x, window='hann')
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(t, x)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Time Domain Signal')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    pos_mask = freqs >= 0
    plt.plot(freqs[pos_mask], np.abs(X[pos_mask]))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Frequency Domain')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 200)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Signal frequency: {f0} Hz")
    print(f"Peak magnitude: {np.max(np.abs(X)):.2f}")
    print(f"Expected magnitude: {A * len(x) / 2:.2f}")


def example_2_multi_frequency():
    """Example 2: Analysis of multi-frequency signal."""
    print("\nExample 2: Multi-frequency Signal Analysis")
    print("=" * 50)
    
    # Generate test signals
    signals = generate_test_signals(fs=1000.0, duration=1.0)
    
    # Analyze multifrequency signal
    analyzer = SpectralAnalyzer(1000.0)
    freqs, X = analyzer.compute_fft(signals['multifreq'], window='hann')
    
    # Plot
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(signals['time'], signals['multifreq'])
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Multi-frequency Signal')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    pos_mask = freqs >= 0
    plt.semilogy(freqs[pos_mask], np.abs(X[pos_mask]))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (log scale)')
    plt.title('Spectrum')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 200)
    
    plt.tight_layout()
    plt.show()
    
    # Find peaks
    from scipy import signal
    peaks, _ = signal.find_peaks(np.abs(X[pos_mask]), height=np.max(np.abs(X[pos_mask])) * 0.1)
    peak_freqs = freqs[pos_mask][peaks]
    peak_mags = np.abs(X[pos_mask])[peaks]
    
    print("Detected frequencies:")
    for freq, mag in zip(peak_freqs, peak_mags):
        print(f"  {freq:.1f} Hz: {mag:.2f}")


def example_3_rotor_faults():
    """Example 3: Rotor fault analysis."""
    print("\nExample 3: Rotor Fault Analysis")
    print("=" * 40)
    
    # Initialize rotor analyzer
    rotor_analyzer = RotordynamicsAnalyzer(fs=1000.0)
    
    # Parameters
    rpm = 1800
    fault_types = ['healthy', 'unbalance', 'misalignment', 'bearing_defect']
    
    # Generate and analyze signals
    plt.figure(figsize=(15, 10))
    
    for i, fault_type in enumerate(fault_types):
        # Generate signal
        signal = rotor_analyzer.generate_rotor_signal(rpm, duration=1.0, fault_type=fault_type)
        
        # Extract features
        features = rotor_analyzer.extract_fault_features(signal, rpm)
        
        # Plot time domain
        plt.subplot(2, 2, i+1)
        t = np.linspace(0, 1.0, len(signal), endpoint=False)
        plt.plot(t, signal)
        plt.title(f'{fault_type.title()} - Time Domain')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        
        # Print key features
        print(f"\n{fault_type.upper()} Features:")
        print(f"  RMS: {features['rms']:.3f}")
        print(f"  Crest Factor: {features['crest_factor']:.3f}")
        print(f"  Kurtosis: {features['kurtosis']:.3f}")
        print(f"  1X Magnitude: {features['1X_magnitude']:.3f}")
        print(f"  2X Magnitude: {features['2X_magnitude']:.3f}")
    
    plt.tight_layout()
    plt.show()


def example_4_machine_learning():
    """Example 4: Machine learning classification."""
    print("\nExample 4: Machine Learning Classification")
    print("=" * 50)
    
    # Generate dataset
    print("Generating rotor dataset...")
    dataset = generate_rotor_dataset(
        rpm_range=(1000, 3000),
        fault_types=['healthy', 'unbalance', 'misalignment', 'bearing_defect'],
        n_samples_per_fault=50
    )
    
    print(f"Dataset size: {len(dataset['signals'])} samples")
    print(f"Fault types: {set(dataset['labels'])}")
    
    # Compare models
    print("\nComparing machine learning models...")
    results = compare_models(dataset['features'], dataset['labels'])
    
    # Print results
    print("\nModel Comparison Results:")
    for model, metrics in results.items():
        print(f"{model}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.3f}")
    
    # Train best model
    print("\nTraining Random Forest classifier...")
    classifier = FaultClassifier('random_forest')
    X, y = classifier.prepare_data(dataset['features'], dataset['labels'])
    train_results = classifier.train(X, y)
    
    print(f"Training Results:")
    for metric, value in train_results.items():
        print(f"  {metric}: {value:.3f}")
    
    # Test on individual signals
    print("\nTesting on individual signals:")
    rotor_analyzer = RotordynamicsAnalyzer(fs=1000.0)
    rpm = 1800
    
    for fault_type in ['healthy', 'unbalance', 'misalignment', 'bearing_defect']:
        signal = rotor_analyzer.generate_rotor_signal(rpm, duration=1.0, fault_type=fault_type)
        features = rotor_analyzer.extract_fault_features(signal, rpm)
        
        X_single = np.array([list(features.values())])
        prediction = classifier.predict(X_single)[0]
        predicted_label = classifier.label_encoder.inverse_transform([prediction])[0]
        
        print(f"True: {fault_type}, Predicted: {predicted_label}")


def example_5_advanced_analysis():
    """Example 5: Advanced spectral analysis."""
    print("\nExample 5: Advanced Spectral Analysis")
    print("=" * 45)
    
    # Generate complex signal
    fs = 1000.0
    duration = 2.0
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    
    # Signal with harmonics and noise
    f0 = 25.0
    x = (np.sin(2 * np.pi * f0 * t) + 
         0.5 * np.sin(2 * np.pi * 2 * f0 * t) + 
         0.3 * np.sin(2 * np.pi * 3 * f0 * t) +
         0.1 * np.random.normal(0, 1, len(t)))
    
    # Analyze with different methods
    analyzer = SpectralAnalyzer(fs)
    
    # FFT analysis
    freqs_fft, X_fft = analyzer.compute_fft(x, window='hann')
    
    # PSD analysis
    freqs_psd, psd = analyzer.compute_psd(x, nperseg=512, window='hann')
    
    # Plot comparison
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(t, x)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Time Domain Signal')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    pos_mask = freqs_fft >= 0
    plt.semilogy(freqs_fft[pos_mask], np.abs(X_fft[pos_mask]))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (log scale)')
    plt.title('FFT Spectrum')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 150)
    
    plt.subplot(1, 3, 3)
    plt.semilogy(freqs_psd, psd)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (log scale)')
    plt.title('Power Spectral Density')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 150)
    
    plt.tight_layout()
    plt.show()
    
    # Bispectrum analysis (simplified)
    print("Computing bispectrum...")
    try:
        f1, f2, bispec = analyzer.compute_bispectrum(x, nfft=128)
        print(f"Bispectrum shape: {bispec.shape}")
        print(f"Frequency range: {f1[0]:.1f} to {f1[-1]:.1f} Hz")
    except Exception as e:
        print(f"Bispectrum computation failed: {e}")


def main():
    """Run all examples."""
    print("Higher Order Spectra (HOS) Analysis Examples")
    print("=" * 50)
    print("This script demonstrates various spectral analysis techniques")
    print("for rotordynamics and fault detection applications.\n")
    
    try:
        example_1_basic_fft()
        example_2_multi_frequency()
        example_3_rotor_faults()
        example_4_machine_learning()
        example_5_advanced_analysis()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("Check the generated plots for visual results.")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


