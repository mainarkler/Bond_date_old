from __future__ import annotations

import argparse
import json
import logging

from services.company_news_analysis import get_company_news_analysis_sync

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Company investment news analyzer")
    parser.add_argument("query", help="Company name or ticker")
    args = parser.parse_args()

    payload = get_company_news_analysis_sync(args.query)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
