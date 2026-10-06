FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /uvx /bin/

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH"

# Cache third-party dependencies separately from application source.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project \
    && useradd --create-home --uid 10001 appuser

COPY src/ src/
RUN uv sync --locked --no-dev --no-editable
COPY app.py ./
COPY .streamlit/config.toml .streamlit/config.toml
USER appuser

EXPOSE 8501
CMD ["sh", "-c", "stackoverflow-db-init && exec streamlit run app.py --server.address=0.0.0.0 --server.port=8501 --server.headless=true --browser.gatherUsageStats=false"]
