# Use Python 3.11-slim base image
FROM python:3.11-slim

# Set environment variable to avoid writing .pyc files
ENV PYTHONDONTWRITEBYTECODE 1

# Set environment variable to buffer output for logs
ENV PYTHONUNBUFFERED 1

# Copy application files to /app directory in the container
COPY ./src /app/src
COPY ./requirements.txt /app/requirements.txt

# Set the working directory to /app
WORKDIR /app

# Install dependencies from requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the port that Uvicorn will run on
ENV PORT=8000
EXPOSE 8000

# Command to start the Uvicorn server with hot-reload
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
