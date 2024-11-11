# Airbnb Data Analysis and Prediction Project

This project performs data analysis and modeling on Airbnb data to predict various features related to room availability and review counts. It includes an interactive API for predictions using a trained model, packaged in a Docker container for easy deployment.

## Table of Contents

  - [Project Overview](#project-overview)
  - [Project Components](#project-components)
  - [Clone the Repository](#clone-the-repository)
  - [Set Up Dependencies](#set-up-dependencies)
  - [Run the FastAPI Application](#run-the-fastapi-application)
  - [API Usage](#api-usage)
  - [Dockerization](#dockerization)
  - [Docker Hub Repo](#docker-hub-repo)


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

     
## Docker Hub Repo

Docker image on Docker Hub [Click Here](https://hub.docker.com/repository/docker/bodaaa/qafza_docker_task/general).
