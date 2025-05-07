from fastapi import APIRouter
from fastapi.responses import JSONResponse


router = APIRouter()

@router.get('/')
def home():
    return JSONResponse({'message': 'Welcome to the FullNameMatcher app!'})