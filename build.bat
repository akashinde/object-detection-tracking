@echo off
REM Build script for detection-tracking-app with OpenAI API key

REM Check if OPENAI_API_KEY is provided
if "%OPENAI_API_KEY%"=="" (
    echo Error: OPENAI_API_KEY environment variable is not set
    echo Please set your OpenAI API key:
    echo set OPENAI_API_KEY=your-api-key-here
    echo.
    echo Or run the build command with the API key:
    echo OPENAI_API_KEY=your-api-key-here build.bat
    pause
    exit /b 1
)

echo Building Docker image with OpenAI API key...
echo API Key: %OPENAI_API_KEY:~0,10%...

REM Build the Docker image
docker build --build-arg OPENAI_API_KEY="%OPENAI_API_KEY%" -t detection-app .

if %ERRORLEVEL% EQU 0 (
    echo ✅ Docker image built successfully!
    echo.
    echo To run the container:
    echo docker run --rm -p 3000:3000 -p 5000:5000 detection-app
    echo.
    echo The application will be available at:
    echo - Frontend: http://localhost:3000
    echo - Backend API: http://localhost:5000
) else (
    echo ❌ Docker build failed!
    pause
    exit /b 1
) 