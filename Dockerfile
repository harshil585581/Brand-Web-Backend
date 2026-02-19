# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app/backend
COPY . backend/

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable
ENV MODULE_NAME="backend.main"
ENV VARIABLE_NAME="app"
ENV PORT="8000"

# Run app.py when the container launches
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
