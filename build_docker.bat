@echo off
echo Building Docker Container...
docker build -t cohort-monitoring .
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Docker build failed. Please ensure Docker Desktop is running.
    echo If you see WSL/Virtualization errors, you may need to enable Virtualization in BIOS.
    pause
    exit /b %errorlevel%
)
echo.
echo [SUCCESS] Build complete! You can now run 'run_docker.bat'
pause
