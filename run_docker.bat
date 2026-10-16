@echo off
echo Running CohortStudio...
echo Open http://localhost:8501 in your browser once it starts.
echo First start: the log below shows a one-time setup token; the login page asks for it to create the administrator.
echo.
if not exist data mkdir data
set ENVFILE=
if exist .env set ENVFILE=--env-file .env
REM Datasets, accounts and traces live in .\data (kept when the container is removed)
docker run --rm --name cohortstudio -p 8501:8501 -v "%cd%\data:/app/data" -v "%cd%\config:/app/config" %ENVFILE% cohortstudio
pause
