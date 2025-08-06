# Mills XGBoost Optimization System

A production-ready system for mill performance prediction and parameter optimization using XGBoost regression models and Bayesian optimization techniques.

## Overview

This project provides:

1. **XGBoost Regression Model**: A robust implementation for predicting mill performance metrics like PSI80 or FR200 based on operational parameters.

2. **Bayesian Optimization**: An optimization framework to tune mill parameters (Ore, WaterMill, etc.) for achieving optimal performance.

3. **API Interface**: FastAPI application to expose model training, prediction, and optimization capabilities.

## Project Structure

```
mills-xgboost/
├── app/
│   ├── database/
│   │   └── db_connector.py   # PostgreSQL database connector
│   ├── models/
│   │   ├── xgboost_model.py  # XGBoost model implementation
│   │   └── data_processor.py # Data preprocessing pipeline
│   ├── optimization/
│   │   └── bayesian_opt.py   # Bayesian optimization module
│   └── api/
│       ├── endpoints.py      # API endpoint definitions
│       └── schemas.py        # Pydantic models for request/response
├── config/
│   └── settings.py           # Configuration settings
├── logs/
│   └── ...                   # Log files
├── models/
│   └── ...                   # Saved models
└── requirements.txt
```

## Features

### Direct PostgreSQL Integration
- Connects directly to the mill sensor and ore quality databases
- Handles proper joining of mill sensor data (1-minute intervals) with ore quality lab data

### XGBoost Regression Model
- Robust model implementation with comprehensive logging
- Support for different target variables (PSI80, FR200)
- Feature engineering and data preprocessing pipelines

### Bayesian Optimization
- Parameter tuning for mill operations
- Configurable parameter bounds
- Support for both maximization and minimization objectives
- Recommendation system for optimal parameter settings

### FastAPI Interface
- RESTful API for model training, prediction, and optimization
- Proper validation using Pydantic schemas
- Comprehensive error handling and logging

## Getting Started

### Prerequisites
- Python 3.8+
- PostgreSQL database with mill sensor and ore quality data

### Installation

1. Clone the repository
2. Install dependencies:
```
pip install -r requirements.txt
```

### Running the API

```
cd mills-xgboost
uvicorn app.main:app --reload
```

The API will be available at http://localhost:8000/

## API Endpoints

### Training a Model
```
POST /api/v1/train
```

### Making Predictions
```
POST /api/v1/predict
```

### Optimizing Parameters
```
POST /api/v1/optimize
```

### Listing Models
```
GET /api/v1/models
```

## Data Processing

The system handles:
- Time series data with 1-minute frequency
- Case-sensitive column names from PostgreSQL
- Proper resampling and joining of mill sensor data with ore quality lab data
- Data smoothing using rolling window averaging

## Deployment

For production deployment, consider:
1. Using environment variables for sensitive information
2. Setting up proper authentication
3. Configuring a production WSGI server
4. Setting up monitoring and logging
