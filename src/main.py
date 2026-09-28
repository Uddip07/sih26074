"""
Gram-Vani application entry point.

    uvicorn src.main:app --port 8080

Serves the forecast API and the dashboard, and (unless AGROMET_AUTO_REFRESH=0) issues today's live forecast in the
background each morning. See src/dashboard/app.py for the endpoints and docs/architecture.md for the system.
"""

from src.dashboard.app import app

__all__ = ["app"]

if __name__ == "__main__":
    import uvicorn

    from src.common.config import load_config

    srv = load_config().dashboard["server"]
    uvicorn.run("src.main:app", host=srv["host"], port=int(srv["port"]))
