from fastapi import APIRouter, HTTPException
from observability.tracing import RetrievalTraceData

router = APIRouter()

# In-memory trace store (replaced by DB in P2.2)
_trace_store: dict[str, dict] = {}


def save_trace(trace: RetrievalTraceData) -> None:
    _trace_store[trace.trace_id] = trace.to_dict()


@router.get("/traces/{trace_id}")
async def get_trace(trace_id: str):
    trace = _trace_store.get(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")
    return trace
