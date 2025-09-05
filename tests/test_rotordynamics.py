#!/usr/bin/env python3
"""
Test suite for rotordynamics module.

This module contains unit tests for the rotordynamics functionality.
"""

import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from rotordynamics import RotordynamicsAnalyzer, generate_rotor_dataset


class TestRotordynamicsAnalyzer(unittest.TestCase):
    """Test cases for RotordynamicsAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.fs = 1000.0
        self.analyzer = RotordynamicsAnalyzer(self.fs)
        
    def test_initialization(self):
        """Test analyzer initialization."""
        self.assertEqual(self.analyzer.fs, self.fs)
        self.assertIsInstance(self.analyzer.fault_features, dict)
        
    def test_generate_rotor_signal_healthy(self):
        """Test generation of healthy rotor signal."""
        rpm = 1800
        duration = 1.0
        
        signal = self.analyzer.generate_rotor_signal(rpm, duration, 'healthy')
        
        # Check output
        self.assertIsInstance(signal, np.ndarray)
        self.assertEqual(len(signal), int(self.fs * duration))
        
        # Healthy signal should have relatively low amplitude
        rms = np.sqrt(np.mean(signal**2))
        self.assertLess(rms, 2.0)  # Reasonable upper bound
        
    def test_generate_rotor_signal_unbalance(self):
        """Test generation of unbalance rotor signal."""
        rpm = 1800
        duration = 1.0
        
        signal = self.analyzer.generate_rotor_signal(rpm, duration, 'unbalance')
        
        # Check output
        self.assertIsInstance(signal, np.ndarray)
        self.assertEqual(len(signal), int(self.fs * duration))
        
        # Unbalance should have higher amplitude than healthy
        rms = np.sqrt(np.mean(signal**2))
        self.assertGreater(rms, 0.5)  # Should be significant
        
    def test_generate_rotor_signal_misalignment(self):
        """Test generation of misalignment rotor signal."""
        rpm = 1800
        duration = 1.0
        
        signal = self.analyzer.generate_rotor_signal(rpm, duration, 'misalignment')
        
        # Check output
        self.assertIsInstance(signal, np.ndarray)
        self.assertEqual(len(signal), int(self.fs * duration))
        
    def test_generate_rotor_signal_bearing_defect(self):
        """Test generation of bearing defect rotor signal."""
        rpm = 1800
        duration = 1.0
        
        signal = self.analyzer.generate_rotor_signal(rpm, duration, 'bearing_defect')
        
        # Check output
        self.assertIsInstance(signal, np.ndarray)
        self.assertEqual(len(signal), int(self.fs * duration))
        
    def test_generate_rotor_signal_invalid_fault(self):
        """Test generation with invalid fault type."""
        rpm = 1800
        duration = 1.0
        
        with self.assertRaises(ValueError):
            self.analyzer.generate_rotor_signal(rpm, duration, 'invalid_fault')
            
    def test_extract_fault_features(self):
        """Test feature extraction."""
        rpm = 1800
        duration = 1.0
        
        # Generate signal
        signal = self.analyzer.generate_rotor_signal(rpm, duration, 'unbalance')
        
        # Extract features
        features = self.analyzer.extract_fault_features(signal, rpm)
        
        # Check that features are extracted
        self.assertIsInstance(features, dict)
        self.assertGreater(len(features), 0)
        
        # Check for expected features
        expected_features = ['rms', 'peak_to_peak', 'crest_factor', 'kurtosis', 
                           'skewness', '1X_magnitude', '2X_magnitude', '3X_magnitude']
        
        for feature in expected_features:
            self.assertIn(feature, features)
            self.assertIsInstance(features[feature], (int, float))
            
        # Check that RMS is positive
        self.assertGreater(features['rms'], 0)
        
        # Check that crest factor is reasonable
        self.assertGreater(features['crest_factor'], 1.0)
        
    def test_classify_fault(self):
        """Test fault classification."""
        # Test with different fault types
        fault_types = ['healthy', 'unbalance', 'misalignment', 'bearing_defect']
        
        for fault_type in fault_types:
            # Generate signal
            signal = self.analyzer.generate_rotor_signal(1800, 1.0, fault_type)
            features = self.analyzer.extract_fault_features(signal, 1800)
            
            # Classify
            predicted = self.analyzer.classify_fault(features)
            
            # Check that classification returns a string
            self.assertIsInstance(predicted, str)
            self.assertIn(predicted, fault_types)


class TestRotorDataset(unittest.TestCase):
    """Test cases for rotor dataset generation."""
    
    def test_generate_rotor_dataset_default(self):
        """Test default dataset generation."""
        dataset = generate_rotor_dataset()
        
        # Check dataset structure
        self.assertIn('signals', dataset)
        self.assertIn('labels', dataset)
        self.assertIn('features', dataset)
        self.assertIn('feature_names', dataset)
        
        # Check data types
        self.assertIsInstance(dataset['signals'], np.ndarray)
        self.assertIsInstance(dataset['labels'], list)
        self.assertIsInstance(dataset['features'], list)
        self.assertIsInstance(dataset['feature_names'], list)
        
        # Check dimensions
        self.assertEqual(len(dataset['signals']), len(dataset['labels']))
        self.assertEqual(len(dataset['features']), len(dataset['labels']))
        
        # Check that we have multiple fault types
        unique_labels = set(dataset['labels'])
        self.assertGreater(len(unique_labels), 1)
        
    def test_generate_rotor_dataset_custom(self):
        """Test custom dataset generation."""
        rpm_range = (1500, 2500)
        fault_types = ['healthy', 'unbalance']
        n_samples = 20
        
        dataset = generate_rotor_dataset(
            rpm_range=rpm_range,
            fault_types=fault_types,
            n_samples_per_fault=n_samples
        )
        
        # Check dimensions
        expected_total = len(fault_types) * n_samples
        self.assertEqual(len(dataset['signals']), expected_total)
        self.assertEqual(len(dataset['labels']), expected_total)
        
        # Check that only specified fault types are present
        unique_labels = set(dataset['labels'])
        self.assertEqual(unique_labels, set(fault_types))
        
        # Check that each fault type has the right number of samples
        for fault_type in fault_types:
            count = dataset['labels'].count(fault_type)
            self.assertEqual(count, n_samples)


class TestIntegration(unittest.TestCase):
    """Integration tests for rotordynamics."""
    
    def test_full_analysis_pipeline(self):
        """Test complete rotordynamics analysis pipeline."""
        # Generate dataset
        dataset = generate_rotor_dataset(
            rpm_range=(1000, 2000),
            fault_types=['healthy', 'unbalance', 'misalignment'],
            n_samples_per_fault=10
        )
        
        # Check that we can extract features for all signals
        analyzer = RotordynamicsAnalyzer(1000.0)
        
        for i, signal in enumerate(dataset['signals']):
            # Get corresponding RPM (simplified - in real scenario this would be stored)
            rpm = 1500  # Use middle of range
            
            # Extract features
            features = analyzer.extract_fault_features(signal, rpm)
            
            # Check that features were extracted
            self.assertIsInstance(features, dict)
            self.assertGreater(len(features), 0)
            
            # Classify fault
            predicted = analyzer.classify_fault(features)
            
            # Check that classification worked
            self.assertIsInstance(predicted, str)
            
    def test_feature_consistency(self):
        """Test that features are consistent across similar signals."""
        analyzer = RotordynamicsAnalyzer(1000.0)
        
        # Generate multiple signals of the same type
        signals = []
        for i in range(5):
            signal = analyzer.generate_rotor_signal(1800, 1.0, 'unbalance')
            signals.append(signal)
        
        # Extract features for all signals
        features_list = []
        for signal in signals:
            features = analyzer.extract_fault_features(signal, 1800)
            features_list.append(features)
        
        # Check that features are reasonably consistent
        # (allowing for some variation due to noise)
        feature_names = features_list[0].keys()
        
        for feature_name in feature_names:
            values = [f[feature_name] for f in features_list]
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            # Coefficient of variation should be reasonable (< 50% for most features)
            if mean_val > 0:
                cv = std_val / mean_val
                self.assertLess(cv, 0.5, f"Feature {feature_name} has high variation: CV = {cv:.3f}")


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)

