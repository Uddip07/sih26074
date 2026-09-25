"""
NASA GPM IMERG Daily Satellite Precipitation Ingestion Module
Bulk downloads NASA GPM IMERG Final Daily (V07B) NetCDF4 files from GES DISC.
Uses standard SessionWithHeaderRedirection for NASA Earthdata URS authentication via ~/.netrc.
"""

import os
import sys
import time
import netrc
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import requests

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

DATA_DIR = os.path.join(BASE_DIR, "data", "raw", "satellite", "imerg")
FAILED_LOG_PATH = os.path.join(DATA_DIR, "failed_downloads.txt")

GES_DISC_BASE_URL = "https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDF.07"
EARTHDATA_HOST = "urs.earthdata.nasa.gov"


class SessionWithHeaderRedirection(requests.Session):
    """
    Subclass of requests.Session that preserves NASA Earthdata authentication
    headers across redirects between GES DISC and URS authentication servers.
    """
    AUTH_HOST = EARTHDATA_HOST

    def __init__(self, username: Optional[str] = None, password: Optional[str] = None):
        super().__init__()
        if username and password:
            self.auth = (username, password)

    def rebuild_auth(self, prepared_request, response):
        headers = prepared_request.headers
        url = prepared_request.url

        if "Authorization" in headers:
            original_parsed = requests.utils.urlparse(response.request.url)
            redirect_parsed = requests.utils.urlparse(url)

            if (
                original_parsed.hostname != redirect_parsed.hostname
                and redirect_parsed.hostname != self.AUTH_HOST
                and original_parsed.hostname != self.AUTH_HOST
            ):
                del headers["Authorization"]
        return


def get_netrc_path() -> Optional[str]:
    """Find .netrc or _netrc in user home directory."""
    home = os.path.expanduser("~")
    for name in [".netrc", "_netrc"]:
        candidate = os.path.join(home, name)
        if os.path.isfile(candidate):
            return candidate
    return None


def load_earthdata_credentials() -> Tuple[Optional[str], Optional[str]]:
    """
    Load Earthdata login credentials from ~/.netrc.
    Returns (username, password) or (None, None) if not configured.
    """
    netrc_file = get_netrc_path()
    if not netrc_file:
        return None, None

    try:
        auth_data = netrc.netrc(netrc_file)
        account = auth_data.authenticators(EARTHDATA_HOST)
        if account:
            return account[0], account[2]
        # Fallback: check if there is a default authenticator
        if auth_data.default:
            return auth_data.default[0], auth_data.default[2]
    except Exception as e:
        print(f"[WARN] Error reading credentials from {netrc_file}: {e}")

    return None, None


def print_netrc_instructions() -> None:
    """Print clear instructions for setting up NASA Earthdata authentication."""
    home = os.path.expanduser("~")
    target_netrc = os.path.join(home, ".netrc")
    print(
        "\n" + "=" * 76 + "\n"
        "[AUTH ERROR] NASA Earthdata Login credentials (~/.netrc) not found!\n\n"
        "To download NASA GPM IMERG satellite precipitation data, follow these steps:\n"
        "1. Register a free NASA Earthdata account at:\n"
        "   https://urs.earthdata.nasa.gov/\n"
        "2. Authorize the 'NASA GESDISC DATA ARCHIVE' application in your profile:\n"
        "   https://urs.earthdata.nasa.gov/users/<your-username>/app_authorizations\n"
        "3. Create your credentials file at:\n"
        f"   {target_netrc}\n"
        "   with the following line:\n"
        "   machine urs.earthdata.nasa.gov login <YOUR_USERNAME> password <YOUR_PASSWORD>\n"
        "4. (Optional on Windows) If permissions error, run: icacls %USERPROFILE%\\.netrc /inheritance:r /grant:r %USERNAME%:(R,W)\n"
        + "=" * 76 + "\n"
    )


def build_imerg_url_and_filename(date_str: str) -> Tuple[str, str]:
    """
    Build GES DISC download URL and standard filename for a given YYYY-MM-DD date.
    URL Format:
    https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDF.07/{YYYY}/{MM}/3B-DAY.MS.MRG.3IMERG.{YYYYMMDD}-S000000-E235959.V07B.nc4
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    yyyy = dt.strftime("%Y")
    mm = dt.strftime("%m")
    yyyymmdd = dt.strftime("%Y%m%d")
    filename = f"3B-DAY.MS.MRG.3IMERG.{yyyymmdd}-S000000-E235959.V07B.nc4"
    url = f"{GES_DISC_BASE_URL}/{yyyy}/{mm}/{filename}"
    return url, filename


def log_download_failure(date_str: str, url: str, reason: str) -> None:
    """Append download failure to failed_downloads.txt log."""
    os.makedirs(DATA_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(FAILED_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] date={date_str} url={url} reason={reason}\n")


def fetch_imerg_data(
    start_date: str = "2024-01-01",
    end_date: str = "2026-08-31",
    output_dir: str = DATA_DIR,
    force_refresh: bool = False,
    request_delay_sec: float = 0.3,
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    """
    Bulk-download NASA GPM IMERG Final Daily (V07B) files from GES DISC.
    
    Parameters:
    - start_date: Beginning date string 'YYYY-MM-DD'
    - end_date: Ending date string 'YYYY-MM-DD'
    - output_dir: Local directory for downloaded NetCDF4 files
    - force_refresh: If True, re-downloads already existing files
    - request_delay_sec: Politeness delay between HTTP calls
    - session: Optional pre-configured requests.Session (for mocking or testing)
    
    Returns:
    - Summary dict containing counts of downloaded, existing, failed, and total files.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Verify credentials if session not explicitly provided
    if session is None:
        username, password = load_earthdata_credentials()
        if not username or not password:
            print_netrc_instructions()
            return {
                "status": "auth_missing",
                "message": "Missing ~/.netrc with urs.earthdata.nasa.gov credentials",
                "downloaded": 0,
                "existing": 0,
                "failed": 0,
                "total": 0,
            }
        session = SessionWithHeaderRedirection(username=username, password=password)

    # 2. Generate date range
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    current_dt = start_dt
    date_list: List[str] = []
    while current_dt <= end_dt:
        date_list.append(current_dt.strftime("%Y-%m-%d"))
        current_dt += timedelta(days=1)

    total_files = len(date_list)
    print(f"NASA GPM IMERG Ingestion: {total_files} days requested ({start_date} to {end_date})")
    print(f"Target directory: {output_dir}")

    downloaded = 0
    existing = 0
    failed = 0

    for idx, d_str in enumerate(date_list, 1):
        url, filename = build_imerg_url_and_filename(d_str)
        target_path = os.path.join(output_dir, filename)

        if os.path.exists(target_path) and os.path.getsize(target_path) > 1000 and not force_refresh:
            existing += 1
            if idx % 50 == 0 or total_files <= 10:
                print(f"  [{idx}/{total_files}] [EXISTS] {filename}")
            continue

        temp_path = target_path + ".tmp"
        try:
            resp = session.get(url, stream=True, timeout=30)
            if resp.status_code == 200:
                # Check for HTML login redirect page disguised as 200
                content_type = resp.headers.get("Content-Type", "")
                if "html" in content_type.lower():
                    log_download_failure(d_str, url, "Received HTML instead of NetCDF binary (auth issue)")
                    print(f"  [{idx}/{total_files}] [AUTH FAIL] {filename} (Redirected to HTML login)")
                    failed += 1
                else:
                    with open(temp_path, "wb") as f_out:
                        for chunk in resp.iter_content(chunk_size=65536):
                            if chunk:
                                f_out.write(chunk)
                    os.replace(temp_path, target_path)
                    file_mb = os.path.getsize(target_path) / (1024 * 1024)
                    print(f"  [{idx}/{total_files}] [DOWNLOADED] {filename} ({file_mb:.2f} MB)")
                    downloaded += 1
            elif resp.status_code == 404:
                # 404 is expected for recent dates (IMERG Final Run has ~3.5 month latency)
                log_download_failure(d_str, url, "HTTP 404 Not Found (latency expected)")
                print(f"  [{idx}/{total_files}] [404 NOT FOUND] {filename}")
                failed += 1
            else:
                log_download_failure(d_str, url, f"HTTP {resp.status_code}")
                print(f"  [{idx}/{total_files}] [HTTP {resp.status_code}] {filename}")
                failed += 1
        except Exception as e:
            log_download_failure(d_str, url, str(e))
            print(f"  [{idx}/{total_files}] [ERROR] {filename}: {e}")
            failed += 1
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

        if request_delay_sec > 0:
            time.sleep(request_delay_sec)

    summary = {
        "status": "completed",
        "start_date": start_date,
        "end_date": end_date,
        "downloaded": downloaded,
        "existing": existing,
        "failed": failed,
        "total": total_files,
    }
    print(f"\nIMERG Fetch Summary: {downloaded} downloaded, {existing} already cached, {failed} failed (total: {total_files})")
    return summary


if __name__ == "__main__":
    # Test range of 5 days as specified
    res = fetch_imerg_data(start_date="2024-01-01", end_date="2024-01-05")
    print("\nResult:", res)
