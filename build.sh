#!/bin/bash

# Build script for detection-tracking-app with OpenAI API key

# Check if OPENAI_API_KEY is provided
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable is not set"
    echo "Please set your OpenAI API key:"
    echo "export OPENAI_API_KEY='your-api-key-here'"
    echo ""
    echo "Or run the build command with the API key:"
    echo "OPENAI_API_KEY='your-api-key-here' ./build.sh"
    exit 1
fi

echo "Building Docker image with OpenAI API key..."
echo "API Key: ${OPENAI_API_KEY:0:10}..." # Show first 10 chars for verification

# Build the Docker image
docker build \
    --build-arg OPENAI_API_KEY="$OPENAI_API_KEY" \
    -t detection-app .

if [ $? -eq 0 ]; then
    echo "✅ Docker image built successfully!"
    echo ""
    echo "To run the container:"
    echo "docker run --rm -p 3000:3000 -p 5000:5000 detection-app"
    echo ""
    echo "The application will be available at:"
    echo "- Frontend: http://localhost:3000"
    echo "- Backend API: http://localhost:5000"
else
    echo "❌ Docker build failed!"
    exit 1
fi 