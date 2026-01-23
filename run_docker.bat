@echo off
echo Running Cohort Monitoring Package...
echo Open http://localhost:8501 in your browser once it starts.
echo.
docker run -p 8501:8501 -e GOOGLE_API_KEY=%GOOGLE_API_KEY% cohort-monitoring
pause
