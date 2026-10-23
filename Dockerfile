# Web app image: docker build -t spricer-web . && docker run -p 8050:8050 spricer-web
FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /usr/local/bin/uv
WORKDIR /srv
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
COPY web ./web
RUN uv sync --locked --no-dev --extra web
ENV PORT=8050
EXPOSE 8050
CMD ["sh", "-c", "uv run --no-sync gunicorn web.app:server --bind 0.0.0.0:${PORT} --workers 2 --timeout 120"]
