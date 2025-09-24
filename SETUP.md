# Local setup with uv

## Install deps

- Install uv
- From this folder: `uv sync -E notebook`

## Register Jupyter kernel

- Create kernel after sync: `uv run python -m ipykernel install --user --name bda --display-name "Python (bda)"`

## Run

- `uv run python scripts/hello.py`
- `uv run jupyter lab`
