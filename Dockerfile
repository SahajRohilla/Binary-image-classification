# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for OpenCV
# We use libgl1 and libosmesa6 to provide the necessary OpenGL support for OpenCV
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY app/ ./app/

# Copy the trained model
# Note: Ensure you have the model generated in models/final_cnn_model.h5
COPY models/ ./models/

# Expose the port the app runs on
EXPOSE 8000

# Set environment variables
ENV MODEL_PATH="models/final_cnn_model.keras"
ENV PYTHONUNBUFFERED=1

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
