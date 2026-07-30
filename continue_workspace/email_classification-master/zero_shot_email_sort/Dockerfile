# Image CPU par défaut — simple et portable.
# Pour du GPU (ex. sur legion, RTX 5070 Ti) : partir d'une base CUDA
# (ex. nvidia/cuda:12.4.1-runtime-ubuntu22.04), installer torch avec le bon
# index (cf. commentaire dans pyproject.toml / uv), et lancer le conteneur
# avec `docker run --gpus all ...` (nvidia-container-toolkit requis sur l'hôte).
FROM python:3.11-slim

WORKDIR /app

# build-essential : requis par certains wheels (tokenizers, sentencepiece...)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# uv : binaire officiel, pas besoin de pip pour l'installer
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Copiés séparément pour profiter du cache Docker : ces couches ne sont
# invalidées que si pyproject.toml/uv.lock changent, pas le code Python.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project

COPY *.py ./
RUN uv sync --locked

# Le venv créé par uv est mis en avant dans le PATH : "python" suffit ensuite,
# pas besoin de préfixer chaque commande avec "uv run".
ENV PATH="/app/.venv/bin:$PATH"
ENV HF_HOME=/data/hf-cache

ENTRYPOINT ["python"]
CMD ["zero_shot_csv.py", "--help"]
