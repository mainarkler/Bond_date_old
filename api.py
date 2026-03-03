import logging

from fastapi import FastAPI, Header, HTTPException

from config import API_TOKEN
from services.report_service import generate_report_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Bond_date_old API", version="1.0.0")


@app.get("/api/report")
def get_report(authorization: str | None = Header(default=None)) -> dict:
    expected = f"Bearer {API_TOKEN}"
    if authorization != expected:
        logger.warning("Unauthorized API access attempt")
        raise HTTPException(status_code=401, detail="Unauthorized")

    report = generate_report_data()
    logger.info("Report delivered via API")
    return report
