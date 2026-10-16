@echo off
echo Starting CohortStudio...

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo Docker is not running. Please start Docker Desktop and try again.
    pause
    exit /b
)

REM Build and Run (Docker Compose v2)
docker compose up --build -d

echo.
echo Application started! Access it at http://localhost:8501
echo First start: run  docker compose logs cohortstudio  to read the one-time setup token for the administrator.
echo To view logs, run: docker compose logs -f
echo To stop, run: docker compose down
pause
