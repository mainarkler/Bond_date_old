import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def generate_report_data() -> dict:
    """Generate structured report data for UI and API consumers."""
    active_view = os.getenv("FORCE_ACTIVE_VIEW", "").strip().lower() or "home"
    report_data = {
        "application": "Bond_date_old",
        "active_view": active_view,
        "supported_views": ["home", "repo", "calendar", "vm", "sell_stres", "index_analytics"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
    }
    logger.info("Report data generated")
    return report_data
