# Higher Order Spectra Analysis for Rotordynamics Fault Detection

## Master's Thesis - Federal University of Uberlândia

This repository contains the complete LaTeX source code for a master's thesis on Higher Order Spectra (HOS) analysis for rotordynamics fault detection.

## Thesis Structure

The thesis is organized into individual chapter files for easy management and collaboration:

### Main Files
- `thesis.tex` - Main thesis file that includes all chapters
- `references.bib` - Bibliography database
- `Makefile` - Compilation automation

### Chapters
- `chapters/abstract.tex` - Abstract
- `chapters/chapter01_introduction.tex` - Introduction
- `chapters/chapter02_literature_review.tex` - Literature Review
- `chapters/chapter03_theoretical_background.tex` - Theoretical Background
- `chapters/chapter04_methodology.tex` - Methodology
- `chapters/chapter05_implementation_results.tex` - Implementation and Results
- `chapters/chapter06_discussion.tex` - Discussion
- `chapters/chapter07_conclusions.tex` - Conclusions and Future Work

### Appendices
- `appendix/appendixA_mathematical_derivations.tex` - Mathematical Derivations
- `appendix/appendixB_code_implementation.tex` - Code Implementation
- `appendix/appendixC_additional_results.tex` - Additional Results
- `appendix/appendixD_dataset_description.tex` - Dataset Description

### Directories
- `figures/` - All figures and images
- `tables/` - Table files (if any)
- `chapters/` - Individual chapter files
- `appendix/` - Appendix files

## Prerequisites

### Required Software
- **LaTeX Distribution**: TeX Live (Linux/macOS) or MiKTeX (Windows)
- **PDF Viewer**: Any PDF viewer for viewing the compiled thesis
- **Text Editor**: Any LaTeX-compatible editor (VS Code, TeXstudio, etc.)

### Required LaTeX Packages
The thesis uses several LaTeX packages. Most are included in standard distributions, but you may need to install:
- `biblatex` or `natbib` for bibliography management
- `pgfplots` for advanced plotting
- `tikz` for diagrams
- `algorithm` and `algorithmic` for algorithm formatting

## Compilation

### Using Makefile (Recommended)

The repository includes a `Makefile` for easy compilation:

```bash
# Compile the complete thesis
make

# Quick compilation (single pass)
make quick

# Clean auxiliary files
make clean

# Clean everything including PDF
make cleanall

# View PDF (macOS)
make view

# View PDF (Linux)
make view-linux

# Check for missing packages
make check

# Show help
make help
```

### Manual Compilation

If you prefer manual compilation:

```bash
# First pass
pdflatex thesis.tex

# Process bibliography
bibtex thesis

# Second pass
pdflatex thesis.tex

# Third pass (for cross-references)
pdflatex thesis.tex
```

## Content Overview

### Chapter 1: Introduction
- Background and motivation
- Problem statement
- Research objectives
- Thesis organization

### Chapter 2: Literature Review
- Traditional vibration analysis methods
- Higher-order spectral analysis
- Applications in fault detection
- Machine learning in fault detection
- Rotordynamics applications

### Chapter 3: Theoretical Background
- Signal processing fundamentals
- Higher-order statistics
- Higher-order spectral analysis
- Rotordynamics theory
- Feature extraction from HOS

### Chapter 4: Methodology
- Overall framework
- Data collection and preprocessing
- Feature extraction
- Feature selection and dimensionality reduction
- Machine learning classification

### Chapter 5: Implementation and Results
- Software implementation
- Experimental setup
- Results and analysis
- Performance evaluation

### Chapter 6: Discussion
- Effectiveness of HOS analysis
- Feature analysis and selection
- Algorithm performance comparison
- Computational considerations
- Comparison with existing methods

### Chapter 7: Conclusions and Future Work
- Research summary
- Contributions
- Limitations and challenges
- Future research directions

## Customization

### Personal Information
Update the following in `thesis.tex`:
- Author name
- University information
- Date (if not using `\today`)

### Content Modification
- Each chapter is in a separate file for easy editing
- Figures should be placed in the `figures/` directory
- Tables can be placed in the `tables/` directory
- Bibliography entries go in `references.bib`

### Styling
- Modify the preamble in `thesis.tex` for different styling
- Adjust page margins, fonts, and formatting as needed
- Customize header and footer styles

## File Organization

```
thesis/
├── thesis.tex                 # Main thesis file
├── references.bib            # Bibliography
├── Makefile                  # Compilation automation
├── README.md                 # This file
├── chapters/                 # Chapter files
│   ├── abstract.tex
│   ├── chapter01_introduction.tex
│   ├── chapter02_literature_review.tex
│   ├── chapter03_theoretical_background.tex
│   ├── chapter04_methodology.tex
│   ├── chapter05_implementation_results.tex
│   ├── chapter06_discussion.tex
│   └── chapter07_conclusions.tex
├── appendix/                 # Appendix files
│   ├── appendixA_mathematical_derivations.tex
│   ├── appendixB_code_implementation.tex
│   ├── appendixC_additional_results.tex
│   └── appendixD_dataset_description.tex
├── figures/                  # Figures and images
├── tables/                   # Table files
└── results/                  # Generated files (PDF, aux files)
```

## Troubleshooting

### Common Issues

1. **Missing Packages**: Use `make check` to identify missing packages
2. **Bibliography Issues**: Ensure `references.bib` is in the same directory
3. **Figure Not Found**: Check file paths in `\includegraphics` commands
4. **Compilation Errors**: Check LaTeX syntax and package compatibility

### Getting Help

- Check LaTeX documentation for specific packages
- Use LaTeX community forums for troubleshooting
- Ensure all required packages are installed

## Contributing

If you're contributing to this thesis:
1. Edit individual chapter files
2. Test compilation with `make quick`
3. Ensure all references are properly cited
4. Update bibliography as needed

## License

This thesis template is provided for academic use. Please ensure compliance with your institution's guidelines and copyright requirements.

## Contact

For questions about this thesis template or content, please contact the author or your academic advisor.
