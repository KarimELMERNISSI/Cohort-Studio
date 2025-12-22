#!/bin/bash

echo "Starting Cohort Monitoring App..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Docker is not running. Please start Docker Desktop and try again."
    exit 1
fi

# Build and Run
docker-compose up --build -d

echo ""
echo "Application started! Access it at http://localhost:8501"
echo "To view logs, run: docker-compose logs -f"
echo "To stop, run: docker-compose down"
