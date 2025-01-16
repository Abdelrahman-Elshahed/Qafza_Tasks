# Airbnb Data Analysis and Prediction Project

This project performs data analysis and modeling on Airbnb data to predict various features related to room availability and review counts. It includes an interactive API for predictions using a trained model, packaged in a Docker container for easy deployment.

## Table of Contents

  - [Project Overview](#project-overview)
  - [Project Components](#project-components)
  - [Clone the Repository](#clone-the-repository)
  - [Set Up Dependencies](#set-up-dependencies)
  - [Run the FastAPI Application](#run-the-fastapi-application)
  - [API Usage](#api-usage)
  - [PostMan API Testing](#postman-api-testing)
  - [Dockerization](#dockerization)
  - [Docker Hub Repo](#docker-hub-repo)
  - [MLflow Integration](#mlflow-integration)


## Project Overview

This project performs data analysis and modeling on Airbnb data to predict various features related to room availability and review counts. It includes an interactive API for predictions using a trained model, packaged in a Docker container for easy deployment.

## Project Components

   - This Jupyter notebook contains the following steps:
     - Data loading and preprocessing
     - Exploratory data analysis (EDA) with visualizations
     - Handling skewness and outliers
     - One-hot encoding and target encoding for categorical features
     - Model training with Random Forest, XGBoost, and LightGBM regressors
     - Saving the best model (XGBoost) as a pickle file (`best_model.pkl`)


## Clone the Repository

   ```bash
   git clone https://github.com/Abdelrahman-Elshahed/Qafza_Tasks.git
   ```


## Set up dependencies:

  Install Python dependencies:
  To install all the required Python packages for the project, run the following command:
  ```bash
  pip install -r requirements.txt
  ```


## Run the FastAPI Application:

  Start the application locally with the following command:
  ```bash
  uvicorn app:app --host 0.0.0.0 --port 8000
  ```


## API Usage

  After starting the API server, you can make POST requests to the /predict endpoint with a JSON payload containing the input features
![Screenshot Response](https://github.com/user-attachments/assets/6204151b-53b7-4f31-bfcb-8793bd3b9ed5)

  
## Dockerization
   - A Docker configuration file to containerize the FastAPI application.
   - Steps:
     - Uses `python:3.10-slim` as the base image.
     - Copies necessary files, installs dependencies, and sets up the API server.
   - Build the Docker image with:
     ```bash
     docker build -t airbnb-prediction .
     ```
   - Run the container with:
     ```bash
     docker run -p 8000:8000 airbnb-prediction
     ```
## PostMan API Testing
  ![Image](https://github.com/user-attachments/assets/e485add4-32ea-4d49-b588-41c65971dced)
## Docker Hub Repo

Docker image on Docker Hub [Click Here](https://hub.docker.com/repository/docker/bodaaa/qafza_docker_task/general).

![dockerhub screenshot](https://github.com/user-attachments/assets/ed1b6c17-877c-416c-ad9e-4455fc238b45)

## MLflow Integration
  
  - MLflow is integrated into the Airbnb Data Analysis and Prediction pipeline for experiment tracking, model versioning, and performance evaluation.
    
![Image](https://github.com/user-attachments/assets/cd5da899-3906-439a-9e17-e09e9e1433fb)

![Image](https://github.com/user-attachments/assets/54d744ad-2554-4cd3-971f-9564909d9de1)

![Image](https://github.com/user-attachments/assets/3161f561-0892-4cb8-81b7-672574511489)
