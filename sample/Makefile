docs: ## Render annotated HTML
	pylit --remove_empty_sections --title_md -t ../../pylit/templates/samples -d html -w labml_samples

pages: ## Copy to lab-ml site
	@cd ../pages; git pull
	cp -r html/* ../pages/


help: ## Show this help.
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'

.PHONY: help docs pages
.DEFAULT_GOAL := help
