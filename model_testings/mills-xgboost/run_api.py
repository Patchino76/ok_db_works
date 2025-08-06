#!/usr/bin/env python
import os
import uvicorn
import argparse
import logging
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/server.log")
    ]
)

logger = logging.getLogger(__name__)

def main():
    """
    Run the Mills XGBoost API server
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Mills XGBoost API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code changes")
    args = parser.parse_args()
    
    # Ensure directories exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("optimization_results", exist_ok=True)
    
    # Log startup information
    logger.info(f"Starting Mills XGBoost API at {args.host}:{args.port}")
    
    # Run the server
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )

if __name__ == "__main__":
    main()
