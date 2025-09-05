#!/usr/bin/env python3
"""
Setup script for Higher Order Spectra (HOS) Analysis package.

Author: Cristofer Antoni Souza Costa
Institution: Federal University of Uberlândia
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="hos-analysis",
    version="1.0.0",
    author="Cristofer Antoni Souza Costa",
    author_email="cristofer.costa@ufu.br",
    description="Higher Order Spectra Analysis for Rotordynamics and Fault Detection",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/HOS",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
            "nbsphinx>=0.8",
        ],
    },
    entry_points={
        "console_scripts": [
            "hos-examples=examples.basic_analysis:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    keywords=[
        "signal processing",
        "spectral analysis",
        "fourier transform",
        "rotordynamics",
        "fault detection",
        "machine learning",
        "vibration analysis",
        "higher order spectra",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/HOS/issues",
        "Source": "https://github.com/yourusername/HOS",
        "Documentation": "https://github.com/yourusername/HOS/docs",
    },
)

