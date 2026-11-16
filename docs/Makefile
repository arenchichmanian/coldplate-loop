# Build the report: `make` in this folder (needs pdflatex + latexmk; TeX Live works).
report.pdf: report.tex ../results/*.png
	latexmk -pdf -interaction=nonstopmode report.tex
clean:
	latexmk -C
