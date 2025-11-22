@echo off
REM FL System Startup Script for Windows

echo 🚀 Starting Federated Learning MLOps System...

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not installed. Please install Docker first.
    exit /b 1
)

REM Check if Docker Compose is installed
docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker Compose is not installed. Please install Docker Compose first.
    exit /b 1
)

REM Prepare client datasets if not exists
if not exist "client_datasets.pkl" (
    echo 📊 Preparing client datasets...
    python save_clients.py
)

REM Build and start all services
echo 🐳 Building and starting containers...
docker-compose up --build -d

REM Wait for services to start
echo ⏳ Waiting for services to start...
timeout /t 30 /nobreak >nul

REM Check service health
echo 🔍 Checking service health...

REM Check API service
curl -s http://localhost:8000/health >nul 2>&1
if errorlevel 1 (
    echo ❌ API Service is not responding
) else (
    echo ✅ API Service is running on http://localhost:8000
)

echo.
echo 🎉 FL MLOps System is ready!
echo.
echo 📱 Access points:
echo    • API Service:  http://localhost:8000
echo    • Dashboard:    http://localhost:8501
echo    • Prometheus:   http://localhost:9090
echo    • Grafana:      http://localhost:3000 (admin/admin)
echo.
echo 📚 API Documentation: http://localhost:8000/docs
echo.
echo To stop the system: docker-compose down
echo To view logs: docker-compose logs -f [service-name]

pause