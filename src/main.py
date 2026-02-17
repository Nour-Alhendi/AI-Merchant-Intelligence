"""
Main entry point for AI Merchant Intelligence System
Production runner.
"""

import logging
import uvicorn


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def main():
    setup_logging()
    logging.info("🚀 Starting AI Merchant Intelligence System...")

    uvicorn.run(
        "api:app",
        host="127.0.0.1",
        port=8001,
        reload=True,
    )


if __name__ == "__main__":
    main()