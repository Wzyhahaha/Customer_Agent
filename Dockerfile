FROM python:3.10-slim

WORKDIR /app

COPY pyproject.toml .
COPY . .

RUN pip install --no-cache-dir -e ".[dev]"

EXPOSE 8000 8501

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
