#!/bin/bash

# FL System Startup Script
echo "🚀 Starting Federated Learning MLOps System..."

# Check if Docker and Docker Compose are installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Prepare client datasets if not exists
if [ ! -f "client_datasets.pkl" ]; then
    echo "📊 Preparing client datasets..."
    python save_clients.py
fi

# Build and start all services
echo "🐳 Building and starting containers..."
docker-compose up --build -d

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 30

# Check service health
echo "🔍 Checking service health..."

# Check API service
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ API Service is running on http://localhost:8000"
else
    echo "❌ API Service is not responding"
fi

# Check Dashboard
if curl -s http://localhost:8501 > /dev/null; then
    echo "✅ Dashboard is running on http://localhost:8501"
else
    echo "❌ Dashboard is not responding"
fi

# Check Prometheus
if curl -s http://localhost:9090 > /dev/null; then
    echo "✅ Prometheus is running on http://localhost:9090"
else
    echo "❌ Prometheus is not responding"
fi

# Check Grafana
if curl -s http://localhost:3000 > /dev/null; then
    echo "✅ Grafana is running on http://localhost:3000 (admin/admin)"
else
    echo "❌ Grafana is not responding"
fi

echo ""
echo "🎉 FL MLOps System is ready!"
echo ""
echo "📱 Access points:"
echo "   • API Service:  http://localhost:8000"
echo "   • Dashboard:    http://localhost:8501"
echo "   • Prometheus:   http://localhost:9090"
echo "   • Grafana:      http://localhost:3000 (admin/admin)"
echo ""
echo "📚 API Documentation: http://localhost:8000/docs"
echo ""
echo "To stop the system: docker-compose down"
echo "To view logs: docker-compose logs -f [service-name]"