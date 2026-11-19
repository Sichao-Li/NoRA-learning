.PHONY: help setup figures

help:
	@printf '%s\n' 'make setup    Install the reader and plotting dependencies' 'make figures  Plot saved results into build/figures/'

setup:
	uv sync

figures:
	uv run python figures/plot_results.py --output build/figures
