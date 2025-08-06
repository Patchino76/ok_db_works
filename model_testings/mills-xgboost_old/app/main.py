from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import os
import time
from datetime import datetime

from config.settings import settings
from app.api.endpoints import router as api_router

# Set up logging
logging_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

os.makedirs(settings.LOGS_DIR, exist_ok=True)
logging.basicConfig(
    level=log_level,
    format=logging_format,
    handlers=[
        logging.FileHandler(os.path.join(settings.LOGS_DIR, f"app_{datetime.now().strftime('%Y%m%d')}.log")),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="API for XGBoost regression models for mill optimization and Bayesian parameter tuning"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response
    except Exception as e:
        logger.error(f"Request failed: {str(e)}")
        process_time = time.time() - start_time
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal server error: {str(e)}"}
        )

# Include API router
app.include_router(api_router, prefix=settings.API_PREFIX)

# Health check endpoint
@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "version": settings.APP_VERSION,
        "timestamp": datetime.now().isoformat()
    }

# Additional info endpoint
@app.get("/info", tags=["info"])
async def app_info():
    """Application information endpoint"""
    return {
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "features": {
            target: features["features"]
            for target, features in settings.FEATURE_SETS.items()
        }
    }

# Error handler for graceful error responses
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.error(f"HTTP error: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

# General exception handler
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
