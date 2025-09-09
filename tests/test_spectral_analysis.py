#!/usr/bin/env python3
"""
Test suite for spectral analysis module.

This module contains unit tests for the spectral analysis functionality.
"""

import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from spectral_analysis import SpectralAnalyzer, generate_test_signals


class TestSpectralAnalyzer(unittest.TestCase):
    """Test cases for SpectralAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.fs = 1000.0
        self.analyzer = SpectralAnalyzer(self.fs)
        
    def test_initialization(self):
        """Test analyzer initialization."""
        self.assertEqual(self.analyzer.fs, self.fs)
        
    def test_fft_basic(self):
        """Test basic FFT computation."""
        # Generate simple sinusoidal signal
        duration = 1.0
        f0 = 50.0
        A = 2.0
        
        t = np.linspace(0, duration, int(self.fs * duration), endpoint=False)
        x = A * np.sin(2 * np.pi * f0 * t)
        
        # Compute FFT
        freqs, X = self.analyzer.compute_fft(x, window='rectangular')
        
        # Check output types and shapes
        self.assertIsInstance(freqs, np.ndarray)
        self.assertIsInstance(X, np.ndarray)
        self.assertEqual(len(freqs), len(x))
        self.assertEqual(len(X), len(x))
        
        # Check that we can find the signal frequency
        pos_mask = freqs >= 0
        magnitude = np.abs(X[pos_mask])
        peak_idx = np.argmax(magnitude)
        detected_freq = freqs[pos_mask][peak_idx]
        
        # Should be close to the input frequency
        self.assertAlmostEqual(detected_freq, f0, delta=1.0)
        
    def test_fft_windowing(self):
        """Test FFT with different windows."""
        # Generate signal
        duration = 1.0
        f0 = 50.0
        t = np.linspace(0, duration, int(self.fs * duration), endpoint=False)
        x = np.sin(2 * np.pi * f0 * t)
        
        windows = ['rectangular', 'hann', 'hamming', 'blackman']
        
        for window in windows:
            freqs, X = self.analyzer.compute_fft(x, window=window)
            
            # Check that FFT was computed
            self.assertIsInstance(X, np.ndarray)
            self.assertEqual(len(X), len(x))
            
    def test_psd_computation(self):
        """Test PSD computation."""
        # Generate signal
        duration = 2.0
        f0 = 50.0
        t = np.linspace(0, duration, int(self.fs * duration), endpoint=False)
        x = np.sin(2 * np.pi * f0 * t) + 0.1 * np.random.normal(0, 1, len(t))
        
        # Compute PSD
        freqs, psd = self.analyzer.compute_psd(x)
        
        # Check output
        self.assertIsInstance(freqs, np.ndarray)
        self.assertIsInstance(psd, np.ndarray)
        self.assertEqual(len(freqs), len(psd))
        
        # PSD should be positive
        self.assertTrue(np.all(psd >= 0))
        
    def test_bispectrum_computation(self):
        """Test bispectrum computation."""
        # Generate simple signal
        duration = 1.0
        t = np.linspace(0, duration, int(self.fs * duration), endpoint=False)
        x = np.sin(2 * np.pi * 25 * t) + 0.5 * np.sin(2 * np.pi * 50 * t)
        
        # Compute bispectrum
        f1, f2, bispec = self.analyzer.compute_bispectrum(x, nfft=64)
        
        # Check output
        self.assertIsInstance(f1, np.ndarray)
        self.assertIsInstance(f2, np.ndarray)
        self.assertIsInstance(bispec, np.ndarray)
        
        # Check shapes
        self.assertEqual(len(f1), 64)
        self.assertEqual(len(f2), 64)
        self.assertEqual(bispec.shape, (64, 64))
        
        # Bispectrum should be non-negative
        self.assertTrue(np.all(bispec >= 0))
        
    def test_trispectrum_computation(self):
        """Test trispectrum computation."""
        # Generate simple signal
        duration = 0.5  # Shorter for computational efficiency
        t = np.linspace(0, duration, int(self.fs * duration), endpoint=False)
        x = np.sin(2 * np.pi * 25 * t)
        
        # Compute trispectrum
        trispec = self.analyzer.compute_trispectrum(x, nfft=32)
        
        # Check output
        self.assertIsInstance(trispec, np.ndarray)
        self.assertEqual(trispec.shape, (32, 32, 32))
        
        # Trispectrum should be non-negative
        self.assertTrue(np.all(trispec >= 0))


class TestTestSignals(unittest.TestCase):
    """Test cases for test signal generation."""
    
    def test_generate_test_signals(self):
        """Test test signal generation."""
        fs = 1000.0
        duration = 1.0
        
        signals = generate_test_signals(fs, duration)
        
        # Check that all expected signals are present
        expected_signals = ['time', 'sinusoid', 'multifreq', 'chirp', 'harmonics', 'noisy']
        
        for signal_name in expected_signals:
            self.assertIn(signal_name, signals)
            self.assertIsInstance(signals[signal_name], np.ndarray)
            
        # Check time vector
        expected_length = int(fs * duration)
        self.assertEqual(len(signals['time']), expected_length)
        
        # Check that time vector is properly spaced
        dt = signals['time'][1] - signals['time'][0]
        expected_dt = 1.0 / fs
        self.assertAlmostEqual(dt, expected_dt, places=10)


class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    def test_full_analysis_pipeline(self):
        """Test complete analysis pipeline."""
        # Generate test signal
        signals = generate_test_signals(1000.0, 1.0)
        x = signals['multifreq']
        
        # Initialize analyzer
        analyzer = SpectralAnalyzer(1000.0)
        
        # Run analysis pipeline
        freqs, X = analyzer.compute_fft(x, window='hann')
        freqs_psd, psd = analyzer.compute_psd(x)
        
        # Check that all steps completed successfully
        self.assertIsNotNone(freqs)
        self.assertIsNotNone(X)
        self.assertIsNotNone(freqs_psd)
        self.assertIsNotNone(psd)
        
        # Check that we can find frequency components
        pos_mask = freqs >= 0
        magnitude = np.abs(X[pos_mask])
        peaks = np.where(magnitude > np.max(magnitude) * 0.1)[0]
        
        # Should find at least one peak
        self.assertGreater(len(peaks), 0)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)


