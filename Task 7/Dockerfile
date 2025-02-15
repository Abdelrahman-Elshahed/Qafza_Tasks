# Base image for building
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install build dependencies and Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ./airbnb ./airbnb
COPY ./models ./models

# Final stage
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /app .

# Set environment variables
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Expose port
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "airbnb/transformers/deployment.py"]