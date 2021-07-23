.PHONY: help clean
.DEFAULT_GOAL := help

clean:
	find . -depth -name "__pycache__" -type d -exec rm -rf {} +
	find . -depth -name ".DS_Store" -type f -exec rm -rf {} +
	find . -depth -name ".ipynb_checkpoints" -type d -exec rm -rf {} +

help: ## Show this help.
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'

