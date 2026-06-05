# Deployment

## Local Development

### Prerequisites

- Python 3.10+
- DashScope API Key
- AMap API Key (optional, for weather/location features)

### Setup

```bash
cp .env.example .env
# Edit .env with your API keys
make install
```

### Run

```bash
# Standalone UI (direct Streamlit)
make run-ui

# Or service mode
make run-api    # Terminal 1: API server at http://localhost:8000
make run-ui     # Terminal 2: UI at http://localhost:8501
```

## Docker

```bash
docker compose up --build
```

Services:
- `api`: FastAPI backend at http://localhost:8000
- `ui`: Streamlit frontend at http://localhost:8501

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `DASHSCOPE_API_KEY` | DashScope LLM API key | Required |
| `AMAP_API_KEY` | AMap geocoding/weather API key | Optional |
| `APP_ENV` | Environment (`local`, `prod`) | `local` |
| `DATABASE_URL` | SQLite/Postgres connection string | `sqlite:///./data/customer_agent.db` |
| `CHROMA_DIR` | Chroma vector store directory | `./chroma_db` |
| `RAG_PIPELINE` | Default RAG pipeline (`baseline` / `enhanced`) | `enhanced` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `MOCK_EXTERNAL_SERVICES` | Mock LLM and external APIs for CI | `false` |

## CI

GitHub Actions runs on every push/PR:
- Install dependencies
- Lint (ruff)
- Tests (pytest with mock services)
