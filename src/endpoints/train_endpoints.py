from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from src.services.matcher import Matcher


router = APIRouter()

@router.post('/train')
def train():
    try:
        matcher = Matcher(mode='train')
        matcher.train()
        return JSONResponse({'message': 'FullNameMatcher is prepared.'})
    except Exception as e:
        raise HTTPException(status_code=404, detail=f'Error: {str(e)}')
