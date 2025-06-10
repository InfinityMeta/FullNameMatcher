from typing import Dict, List

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from src.services.matcher import Matcher


router = APIRouter()

@router.post('/match')
def match(candidates: List[Dict[str, str]], matches_num: int, use_threshold: bool):
    try:
        matcher = Matcher(mode='eval')
        matches = matcher.match(
            candidates=candidates,
            matches_num=matches_num,
            use_threshold=use_threshold
        )
        return JSONResponse(matches)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f'Error: {str(e)}')