.PHONY: help install test lint format check setup clean

help:
	@echo "AI Forex System - Available commands:"
	@echo "  make install  - Install dependencies"
	@echo "  make test     - Run tests"
	@echo "  make lint     - Run linter"
	@echo "  make format   - Format code"
	@echo "  make check    - Run all checks"
	@echo "  make setup    - Setup development environment"
	@echo "  make clean    - Clean up generated files"

setup:
	python3 -m venv venv
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && pip install -r requirements.txt
	. venv/bin/activate && pip install -e .

install: setup
	@echo "Installation complete"

test:
	. venv/bin/activate && pytest tests/ -v

lint:
	. venv/bin/activate && flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	. venv/bin/activate && flake8 . --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics

format:
	. venv/bin/activate && black .

type-check:
	. venv/bin/activate && mypy .

check: lint type-check test

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "build" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache htmlcov .coverage
