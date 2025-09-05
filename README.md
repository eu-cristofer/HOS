# Higher Order Spectra (HOS) Analysis

A comprehensive Python package for advanced spectral analysis with applications in rotordynamics and fault detection, developed as part of master's studies at Federal University of Uberlândia.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Features](#features)
6. [Study Plan](#study-plan)
7. [Documentation](#documentation)
8. [Examples](#examples)
9. [Contributing](#contributing)
10. [License](#license)

## Project Overview

This project aims to support a comprehensive literature review and practical study in the field of signals and systems, with the following main objectives:

- **Survey advanced spectral analysis techniques** such as bispectrum, trispectrum, and Fast Fourier Transform (FFT) as applied to fault detection in rotordynamics.
- **Review machine learning methods** for signal classification, especially for distinguishing fault conditions.
- **Compare methodologies and results** with state-of-the-art approaches reported in the literature.

## Repository Structure

```
HOS/
├── src/                          # Source code modules
│   ├── __init__.py              # Package initialization
│   ├── spectral_analysis.py     # Core spectral analysis tools
│   ├── rotordynamics.py         # Rotordynamics-specific analysis
│   └── ml_classifier.py         # Machine learning classification
├── notebooks/                    # Jupyter notebooks for learning
│   ├── 01_FFT_Fundamentals.ipynb
│   └── 02_Rotordynamics_Analysis.ipynb
├── examples/                     # Practical examples and demos
│   └── basic_analysis.py        # Basic analysis examples
├── docs/                         # Documentation and study materials
│   └── study_guide.md           # Comprehensive study guide
├── data/                         # Datasets and experimental data
├── results/                      # Analysis outputs and figures
├── essays/                       # Academic papers and literature reviews
│   └── review_FT.tex            # Fourier Transform literature review
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/HOS.git
cd HOS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify installation:
```bash
python examples/basic_analysis.py
```

## Quick Start

```python
import sys
import os
sys.path.append('src')

from spectral_analysis import SpectralAnalyzer
from rotordynamics import RotordynamicsAnalyzer

# Basic spectral analysis
analyzer = SpectralAnalyzer(fs=1000.0)
freqs, X = analyzer.compute_fft(signal, window='hann')

# Rotor fault analysis
rotor_analyzer = RotordynamicsAnalyzer(fs=1000.0)
signal = rotor_analyzer.generate_rotor_signal(rpm=1800, fault_type='unbalance')
features = rotor_analyzer.extract_fault_features(signal, rpm=1800)
```

## Features

### Core Spectral Analysis
- **Fast Fourier Transform (FFT)** with proper scaling and normalization
- **Power Spectral Density (PSD)** using Welch's method
- **Bispectrum and Trispectrum** computation for higher-order analysis
- **Windowing functions** (Hann, Hamming, Blackman, etc.)
- **Spectral leakage** analysis and mitigation

### Rotordynamics Applications
- **Fault simulation** for common rotor problems (unbalance, misalignment, bearing defects)
- **Feature extraction** from vibration signals
- **Rotor orbit** analysis and visualization
- **Harmonic analysis** with automatic peak detection
- **Statistical features** (RMS, crest factor, kurtosis, etc.)

### Machine Learning
- **Fault classification** using multiple algorithms (Random Forest, SVM, Neural Networks)
- **Feature selection** and importance analysis
- **Model comparison** and validation
- **Cross-validation** and performance metrics
- **Confusion matrix** visualization

### Educational Resources
- **Interactive Jupyter notebooks** for hands-on learning
- **Comprehensive study guide** with structured learning path
- **Practical examples** with real-world applications
- **Literature review** materials and references

## Study Plan

The project follows a structured 12-week learning path covering fundamentals to advanced applications:

### Phase 1: Fundamentals (Weeks 1-4)
- Mathematical foundations of Fourier analysis
- Discrete Fourier Transform and FFT
- Power Spectral Density estimation
- Higher-order statistics and spectra

### Phase 2: Rotordynamics Applications (Weeks 5-8)
- Rotor dynamics fundamentals
- Common fault types and signatures
- Feature extraction techniques
- Machine learning applications

### Phase 3: Advanced Topics (Weeks 9-12)
- Time-frequency analysis
- Nonlinear analysis methods
- Real data analysis
- System integration

See [docs/study_guide.md](docs/study_guide.md) for detailed learning objectives and weekly schedules.

## Documentation

- **[Study Guide](docs/study_guide.md)**: Comprehensive 12-week learning path
- **[Literature Review](essays/review_FT.tex)**: Mathematical foundations of Fourier analysis
- **[API Documentation](src/)**: Source code documentation and examples

## Examples

### Basic Analysis
```bash
python examples/basic_analysis.py
```

### Interactive Learning
```bash
jupyter notebook notebooks/01_FFT_Fundamentals.ipynb
jupyter notebook notebooks/02_Rotordynamics_Analysis.ipynb
```

### Custom Analysis
```python
from src.spectral_analysis import SpectralAnalyzer
from src.rotordynamics import RotordynamicsAnalyzer
from src.ml_classifier import FaultClassifier

# Your custom analysis code here
```

## Literature Review Topics

### 1. Fundamentals of Signals and Systems
- Review textbooks and key papers on signals, systems, and their mathematical foundations.
- Summarize key concepts: time and frequency domain analysis, linear and nonlinear system behavior.

### 2. Spectral Analysis Techniques
- Study the theory and application of FFT, bispectrum, trispectrum, and related higher-order spectral methods.
- Compare the strengths, limitations, and implementation details of each method.
- Document findings from influential publications and textbooks.

### 3. Feature Extraction and Fault Diagnosis
- Survey approaches for extracting features from spectral analysis, with emphasis on fault detection in rotordynamic systems.
- Review case studies and experiments from the literature.
- Summarize common signal characteristics associated with different fault types.

### 4. Machine Learning for Signal Classification
- Explore supervised and unsupervised learning techniques for classifying faults using extracted features.
- Compare algorithms such as SVM, Random Forest, Neural Networks, and others.
- Analyze validation methods, metrics, and benchmarking strategies as presented in the literature.

### 5. Comparative Analysis
- Synthesize methodologies and results from reviewed papers and reports.
- Identify trends, gaps, and opportunities for improvement in the field.
- Prepare comparative tables and visualizations to summarize literature findings.

### 6. Practical Implementation and Experimentation
- Translate literature insights into practical experiments using simulation and real data.
- Validate and compare results with those reported in the literature.

### 7. Documentation and Reporting
- Maintain detailed notes and summaries of reviewed articles and books.
- Update repository with annotated bibliographies, summaries, and code experiments.

### 8. Resources & Recommended Readings
- List foundational textbooks, review articles, and open datasets relevant to the field.
- Example resources:
  - "Signals and Systems" by Oppenheim & Willsky
  - "The Scientist and Engineer's Guide to Digital Signal Processing" by Steven W. Smith
  - Key journal articles and conference proceedings

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository** and create a feature branch
2. **Follow the coding style** and add appropriate documentation
3. **Add tests** for new functionality
4. **Update documentation** as needed
5. **Submit a pull request** with a clear description

### Development Setup
```bash
git clone https://github.com/yourusername/HOS.git
cd HOS
pip install -r requirements.txt
pip install -e .  # Install in development mode
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings for all functions and classes
- Include examples in docstrings

## License