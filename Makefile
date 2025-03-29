.PHONY: setup transform test clean ingest all

setup:
	uv pip install -e .

transform:
	python bin/transform.py

transform-reserves:
	python bin/transform.py --transforms reserves

test:
	pytest

clean:
	rm -rf data/processed/*
	rm -rf outputs/*
	rm -rf logs/*

ingest:
	npx repomix --ignore="ArgGIS/data/**,**/*.ipynb,**/*.md,**/*.json,**/*.csv,**/*.txt,**/*.yaml,**/*.yml"

all: setup transform