# Use the official Python image as a base
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements to the container and install dependencies
COPY requirements.txt requirements.txt
COPY best_model.pkl /app/

RUN pip install -r requirements.txt

# Copy the entire application code into the container
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Define the command to run the API server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
