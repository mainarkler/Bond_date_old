import logging
import secrets
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from config import API_TOKEN
from services.report_service import generate_report_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Bond_date_old API", version="1.0.0")
security_scheme = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security_scheme)) -> str:
    token = credentials.credentials
    if not secrets.compare_digest(token, API_TOKEN):
        logger.warning("Unauthorized API access attempt")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token

# Используем обычный def, чтобы FastAPI сам вынес вычисления в отдельный поток
@app.get("/api/report")
def get_report(token: str = Depends(verify_token)) -> dict:
    report = generate_report_data()
    logger.info("Report delivered via API")
    return report
