# Structured Products Pricer

[![CI](https://github.com/maxenceherve/structured-products-pricer/actions/workflows/ci.yml/badge.svg)](https://github.com/maxenceherve/structured-products-pricer/actions/workflows/ci.yml)

Pricing, Greeks and brick-by-brick decomposition of vanilla option strategies and structured
products, written from the textbook formulas (Hull, Wilmott, Glasserman, Haug), with an
interactive Streamlit app.

> **Status: work in progress.** Nothing in this repository is investment advice.

## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/maxenceherve/structured-products-pricer.git
cd structured-products-pricer
uv sync --all-extras
```

## Run the app

```bash
uv run streamlit run app/streamlit_app.py
```

## Run the checks

```bash
uv run pytest
uv run ruff check
uv run ruff format --check
uv run mypy
```

## License

[MIT](LICENSE)
