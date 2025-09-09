# =============================================================================
# MAKEFILE FOR THESIS COMPILATION
# =============================================================================

# Variables
MAIN = thesis
TEX = $(MAIN).tex
PDF = $(MAIN).pdf
BIB = references.bib

# LaTeX compiler
LATEX = pdflatex
BIBTEX = bibtex

# Compilation flags
LATEX_FLAGS = -interaction=nonstopmode -file-line-error

# Default target
all: $(PDF)

# Main compilation target
$(PDF): $(TEX) $(BIB)
	@echo "Compiling thesis..."
	$(LATEX) $(LATEX_FLAGS) $(TEX)
	$(BIBTEX) $(MAIN)
	$(LATEX) $(LATEX_FLAGS) $(TEX)
	$(LATEX) $(LATEX_FLAGS) $(TEX)
	@echo "Thesis compiled successfully!"

# Quick compilation (single pass)
quick: $(TEX)
	@echo "Quick compilation..."
	$(LATEX) $(LATEX_FLAGS) $(TEX)
	@echo "Quick compilation completed!"

# Clean auxiliary files
clean:
	@echo "Cleaning auxiliary files..."
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.lol *.fdb_latexmk *.fls *.synctex.gz
	@echo "Clean completed!"

# Clean everything including PDF
cleanall: clean
	@echo "Cleaning everything..."
	rm -f $(PDF)
	@echo "Clean all completed!"

# View PDF (macOS)
view: $(PDF)
	@echo "Opening PDF..."
	open $(PDF)

# View PDF (Linux)
view-linux: $(PDF)
	@echo "Opening PDF..."
	xdg-open $(PDF)

# Check for missing packages
check:
	@echo "Checking for missing packages..."
	$(LATEX) $(LATEX_FLAGS) $(TEX) 2>&1 | grep -i "package" || echo "No missing packages found"

# Help
help:
	@echo "Available targets:"
	@echo "  all        - Compile thesis (default)"
	@echo "  quick      - Quick compilation (single pass)"
	@echo "  clean      - Remove auxiliary files"
	@echo "  cleanall   - Remove all generated files"
	@echo "  view       - Open PDF (macOS)"
	@echo "  view-linux - Open PDF (Linux)"
	@echo "  check      - Check for missing packages"
	@echo "  help       - Show this help"

# Phony targets
.PHONY: all quick clean cleanall view view-linux check help
