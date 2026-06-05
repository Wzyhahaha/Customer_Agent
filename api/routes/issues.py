from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.get("/issues/{issue_id}")
async def get_issue(issue_id: str):
    raise HTTPException(status_code=501, detail="Issue persistence not yet implemented")
