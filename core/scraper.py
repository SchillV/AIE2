import requests
from bs4 import BeautifulSoup
import csv
import sys
from pathlib import Path

URL = "https://www.cursbnr.ro/curs-valutar-bnr"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Content-Type": "application/x-www-form-urlencoded",
    "Referer": URL,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

PAYLOAD = {
    "currency": "IDR",
    "dataStart": "22/02/2020",
}

OUTPUT_FILE = Path("resources") / "data" / "idr_exchange_rates.csv"


def fetch_exchange_rates() -> tuple[list[str], list[list[str]]]:
    print(f"POSTing to {URL} for IDR rates starting 22/02/2020 ...")

    session = requests.Session()
    response = session.post(URL, data=PAYLOAD, headers=HEADERS, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    table = soup.find("table", {"id": "table-currencies"})
    if not table:
        raise ValueError(
            "Table with id='table-currencies' not found in the response. "
            "The site structure may have changed."
        )

    thead = table.find("thead")
    if not thead:
        raise ValueError("<thead> not found inside #table-currencies.")

    col_headers = [th.get_text(strip=True) for th in thead.find_all("th")]

    rows: list[list[str]] = []
    tbody = table.find("tbody")
    source = tbody if tbody else table
    for tr in source.find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if cells:
            rows.append(cells)

    return col_headers, rows


def save_to_csv(col_headers: list[str], rows: list[list[str]], filename: str) -> None:
    # Pad header if some rows have more columns than the <thead> declared
    if rows:
        max_cols = max(len(row) for row in rows)
        while len(col_headers) < max_cols:
            col_headers = col_headers + [f"col_{len(col_headers) + 1}"]

    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(col_headers)
        writer.writerows(rows)
    print(f"Saved {len(rows)} record(s) to '{filename}'.")


def main() -> None:
    try:
        col_headers, rows = fetch_exchange_rates()

        if not rows:
            print("The table was found but contained no data rows.")
            sys.exit(1)

        print(f"Columns : {col_headers}")
        print(f"Records : {len(rows)}")

        save_to_csv(col_headers, rows, OUTPUT_FILE)

    except requests.exceptions.HTTPError as exc:
        print(f"[HTTP error] {exc}")
        sys.exit(1)
    except requests.exceptions.ConnectionError as exc:
        print(f"[Connection error] {exc}")
        sys.exit(1)
    except requests.exceptions.Timeout:
        print("[Timeout] The request took too long.")
        sys.exit(1)
    except requests.exceptions.RequestException as exc:
        print(f"[Request failed] {exc}")
        sys.exit(1)
    except ValueError as exc:
        print(f"[Parse error] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
