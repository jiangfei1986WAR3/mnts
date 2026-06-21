from __future__ import annotations

import argparse
import csv
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List


BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download BTCUSDT 15m klines from Binance.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="15m")
    parser.add_argument("--months", type=int, default=12)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument(
        "--output",
        default="data/btcusdt_15m_1y.csv",
        help="CSV path under current workspace.",
    )
    parser.add_argument("--sleep-seconds", type=float, default=0.15)
    return parser.parse_args()


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int, limit: int) -> List[list]:
    query = urllib.parse.urlencode(
        {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": limit,
        }
    )
    url = f"{BINANCE_KLINES_URL}?{query}"
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = response.read().decode("utf-8")
    import json

    return json.loads(payload)


def ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def main() -> None:
    args = parse_args()

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=30 * args.months)
    end = now

    rows: List[list] = []
    cursor = ms(start)
    end_ms = ms(end)

    while cursor < end_ms:
        batch = fetch_klines(args.symbol, args.interval, cursor, end_ms, args.limit)
        if not batch:
            break
        rows.extend(batch)

        last_open_time = int(batch[-1][0])
        if last_open_time <= cursor:
            break

        cursor = last_open_time + 1
        time.sleep(args.sleep_seconds)

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_asset_volume",
                "number_of_trades",
                "taker_buy_base_asset_volume",
                "taker_buy_quote_asset_volume",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    datetime.fromtimestamp(row[0] / 1000, tz=timezone.utc).isoformat(),
                    row[1],
                    row[2],
                    row[3],
                    row[4],
                    row[5],
                    datetime.fromtimestamp(row[6] / 1000, tz=timezone.utc).isoformat(),
                    row[7],
                    row[8],
                    row[9],
                    row[10],
                ]
            )

    print(f"Saved {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()

