# Airbnb Price Predictor Using Mage AI
A machine learning-based application pipeline that built with Mage.ai to predict Airbnb listing prices using multiple models including XGBoost, LightGBM, and Random Forest.

## Table of Contents

  - [Overview](#overview)
  - [Project Structure](#project-structure)
  - [Features](#features)
  - [Setup](#setup)
  - [Pipeline Structure in Mage AI](#pipeline-structure-in-mage-ai)
  - [Run with Streamlit Application](#run-with-streamlit-application)
  - [Dockerization](#dockerization)
  - [MLflow Integration](#mlflow-integration)
  - [DagsHub Integration with MLflow](#dagshub-integration-with-mlflow)

## Overview

This project implements an end-to-end machine learning pipeline for predicting Airbnb listing prices based on various features. It ingests data from an external API [Click Here](https://raw.githubusercontent.com/Abdelrahman-Elshahed/Qafza_Tasks/refs/heads/main/Task%202/listings.csv), uses Mage.ai tool for orchestration and includes data ingestion, feature engineering, model training, and a Streamlit-based web interface.

## Project Structure

   ```bash
   📦 Airbnb Data Analysis
│
├───📁 .file_versions
│   ├───📁 data_loaders
│   │   └───📜 ingest_data.py
│   │
│   ├───📁 pipelines
│   │   └───📁 pipeline_airbnb
│   │       └───📜 metadata.yaml
│   │
│   ├───📜 requirements.txt
│   │
│   └───📁 transformers
│       ├───📜 airbnb_data_deployment.py
│       │
│       ├───📜 feature_engineering.py
│       │
│       └───📜 model_training.py
│
├───📁 .ssh_tunnel
│       📜 aws_emr.json
│
├───📁 charts
│   │   📜 __init__.py
│   │
│   └───📁 __pycache__
│           📜 __init__.cpython-310.pyc
│
├───📁 custom
│   │   📜 __init__.py
│   │
│   └───📁 __pycache__
│           📜 __init__.cpython-310.pyc
│
├───📁 data_exporters
│   │   📜 __init__.py
│   │
│   └───📁 __pycache__
│           📜 export_titanic_clean.cpython-310.pyc
│           📜 __init__.cpython-310.pyc
│
├───📁 data_loaders
│   │   📜 ingest_data.py
│   │   📜 __init__.py
│   │
│   └───📁 __pycache__
│           📜 ingest_data.cpython-310.pyc
│           📜 load_titanic.cpython-310.pyc
│           📜 __init__.cpython-310.pyc
│
├───📁 dbt
│       📜 profiles.yml
│
├───📁 extensions
│   │   📜 __init__.py
│   │
│   └───📁 __pycache__
│           📜 __init__.cpython-310.pyc
│
├───📁 interactions
│   │   📜 __init__.py
│   │
│   └───📁 __pycache__
│           📜 __init__.cpython-310.pyc
│
├───📁 models
│       📜 lightgbm_model.pkl
│       📜 random_forest_model.pkl
│       📜 xgboost_model.pkl
│
├───📁 pipelines
│   │   📜 __init__.py
│   │
│   ├───📁 pipeline_airbnb
│   │   │   📜 metadata.yaml
│   │   │   📜 __init__.py
│   │   │
│   │   └───📁 __pycache__
│   │           📜 __init__.cpython-310.pyc
│   │
│   └───📁 __pycache__
│           📜 __init__.cpython-310.pyc
│
├───📁 scratchpads
│   │   📜 __init__.py
│   │
│   └───📁 __pycache__
│           📜 __init__.cpython-310.pyc
│
├───📁 transformers
│   │   📜 airbnb_data_deployment.py
│   │   📜 feature_engineering.py
│   │   📜 model_training.py
│   │   📜 __init__.py
│   │
│   └───📁 __pycache__
│           📜 airbnb_data_deployment.cpython-310.pyc
│           📜 deployment.cpython-310.pyc
│           📜 feature_engineering.cpython-310.pyc
│           📜 fill_in_missing_values.cpython-310.pyc
│           📜 model_training.cpython-310.pyc
│           📜 streamlit_app.cpython-310.pyc
│           📜 __init__.cpython-310.pyc
│
├───📁 utils
│   │   📜 __init__.py
│   │
│   └───📁 __pycache__
│           📜 __init__.cpython-310.pyc
│
└───📁 __pycache__
│       📜 __init__.cpython-310.pyc
│
│   📜 .dockerignore
│   📜 .gitignore
│   📜 docker-compose.yml
│   📜 Dockerfile
│   📜 io_config.yaml
│   📜 metadata.yaml
│   📜 requirements.txt
│   📜 __init__.py
   ```

## Features

   - Data ingestion from external API
   - Automated feature engineering
   - Data pipeline using Mage.ai
   - Multiple ML models:
     - Random Forest
     - XGBoost
     - LightGBM
   - Streamlit web interface
   - Containerized deployment
   - Automated testing


## Setup

- Clone the Repository

   ```bash
   git clone https://github.com/Abdelrahman-Elshahed/Qafza_Tasks.git
   ```
- Create and activate a virtual environment:
  ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
  ```
- Set up dependencies

  Install Python dependencies:
  To install all the required Python packages for the project, run the following command:
  ```bash
  pip install -r requirements.txt
  ```
## Pipeline Structure in Mage AI

![Image](https://github.com/user-attachments/assets/a7e3f38a-00f4-4018-ad82-afd6640c132c)

The project uses Mage.ai for pipeline orchestration with the following components:

- Data Ingestion (data_loaders/ingest_data.py)
  - Loads data from external API
  - Performs initial data validation

- Feature Engineering (transformers/feature_engineering.py)

  - Data cleaning
  - Feature creation
  - Encoding categorical variables
  
- Model Training (transformers/model_training.py)

  - Trains multiple models
  - Performs model evaluation
  - Saves trained models
  
 - Model Deployment (transformers/airbnb_data_deployment.py)

  - Streamlit web interface
  - Real-time predictions
  - Model serving

## Dockerization

   - Build the Docker image with:
     ```bash
     docker build -t airbnb-price-predictor .
     ```
   - Run the container with:
     ```bash
     docker run -p 8000:8000 airbnb-price-predictor
     ```
     
![Image](https://github.com/user-attachments/assets/6932d7ff-2a26-425e-8cc7-357e4808968b)


## Run with Streamlit Application

![Image](https://github.com/user-attachments/assets/76e1f6e7-f1a9-48c4-8e40-c1fe9e6b0a2d)



## MLflow Integration
  
  - MLflow is integrated into the Airbnb Price Predictor pipeline for experiment tracking, model versioning, and performance evaluation.

![Image](https://github.com/user-attachments/assets/ac30fa92-dc50-4b4b-9990-b67a623e5ebf)


![Image](https://github.com/user-attachments/assets/fcf61419-ad3f-42da-9bd8-df4964b651e8)


## DagsHub Integration with MLflow
  - DagsHub integrates with MLflow, providing an online, collaborative platform for everyone to track experiments, version models, and manage machine learning projects seamlessly.
  - For DagsHub Experiments [Click Here](https://dagshub.com/Abdelrahman-Elshahed/Qafza_Tasks/experiments).
![Image](https://github.com/user-attachments/assets/890e45c1-af87-4878-8d2e-98ee97618859)

![Image](https://github.com/user-attachments/assets/2da0cea6-c743-4f65-bf32-533c4d6bd285)
