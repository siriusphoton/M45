import argparse
import asyncio
from datetime import date, datetime

from write_daily_log import write_daily_log

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write a daily log in m45_trial.")
    parser.add_argument(
        "processing_day", type=date.fromisoformat, help="Processing day: YYYY-MM-DD"
    )
    parser.add_argument(
        "--as-of",
        type=datetime.fromisoformat,
        required=True,
        help="Exclusive review cutoff, including timezone offset",
    )
    arguments = parser.parse_args()
    asyncio.run(
        write_daily_log(
            arguments.processing_day,
            arguments.as_of,
            expected_database="m45_trial",
        )
    )
