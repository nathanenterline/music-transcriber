#!/bin/bash

echo "🚀 Starting deployment for Music AI..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Install Docker Desktop and try again."
    exit 1
fi

# Create necessary folders
mkdir -p audio_input output logs notebooks

# Build the Docker image
docker-compose build music-ai-app

# Start the app container
docker-compose up -d music-ai-app

echo "✅ Music AI app is now running!"
echo "💡 You can test the app using:"
echo "   docker-compose exec music-ai-app python music_ai.py"
