#!/usr/bin/env python3
"""
Скрипт для запуску backend сервера
"""
import os
import uvicorn

import asyncio
import sys

if __name__ == "__main__":
    # WORKAROUND: Force SelectorEventLoop on Windows to avoid "ConnectionResetError" / "WinError 10054"
    # when serving large static files via ProactorEventLoop (default in Python 3.8+).
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # IMPORTANT (Windows): reload spawns an extra process and can easily double RAM usage.
    # Default: reload disabled. Enable only when actively developing the backend.
    reload = (os.getenv("UVICORN_RELOAD") or "0").lower() in ("1", "true", "yes")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=reload,
        reload_excludes=["output", "cache", "*/__pycache__/*"],
        # loop="asyncio", # Policy set above overrides this behavior effectively for the main process
        log_level="info"
    )

