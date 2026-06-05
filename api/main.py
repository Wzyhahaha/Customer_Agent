from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import chat, health, issues, traces

app = FastAPI(
    title="Customer Agent API",
    description="Traceable RAG customer support agent platform",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["health"])
app.include_router(chat.router, tags=["chat"])
app.include_router(issues.router, tags=["issues"])
app.include_router(traces.router, tags=["traces"])


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )
