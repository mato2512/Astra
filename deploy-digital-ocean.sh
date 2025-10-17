#!/bin/bash

# Astra AI Deployment Script for Digital Ocean
# Run this on your Digital Ocean droplet

set -e

echo "🚀 Starting Astra AI Deployment..."

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
REPO_URL="https://github.com/mato2512/Astra.git"
APP_DIR="/opt/astra"
DOCKER_IMAGE="astra-ai"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker compose &> /dev/null; then
    echo -e "${RED}❌ Docker Compose is not installed. Please install Docker Compose first.${NC}"
    exit 1
fi

# Create app directory if it doesn't exist
echo -e "${YELLOW}📁 Setting up application directory...${NC}"
sudo mkdir -p $APP_DIR
cd $APP_DIR

# Clone or pull latest code
if [ -d ".git" ]; then
    echo -e "${YELLOW}🔄 Pulling latest changes from GitHub...${NC}"
    git pull origin main
else
    echo -e "${YELLOW}📥 Cloning repository from GitHub...${NC}"
    git clone $REPO_URL .
fi

# Stop existing containers
echo -e "${YELLOW}🛑 Stopping existing containers...${NC}"
docker compose down || true

# Clean up old images (optional)
echo -e "${YELLOW}🧹 Cleaning up old Docker images...${NC}"
docker image prune -f || true

# Build with increased memory limits
echo -e "${YELLOW}🔨 Building Docker image...${NC}"
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

# Build the image with proper limits
docker compose build --no-cache \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    --progress=plain 2>&1 | tee build.log

# Check if build was successful
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo -e "${GREEN}✅ Build successful!${NC}"
else
    echo -e "${RED}❌ Build failed! Check build.log for details${NC}"
    tail -n 50 build.log
    exit 1
fi

# Start the containers
echo -e "${YELLOW}🚀 Starting containers...${NC}"
docker compose up -d

# Wait for services to be ready
echo -e "${YELLOW}⏳ Waiting for services to start...${NC}"
sleep 10

# Check container status
echo -e "${YELLOW}📊 Container Status:${NC}"
docker compose ps

# Show logs
echo -e "${YELLOW}📝 Recent logs:${NC}"
docker compose logs --tail=50

# Check if service is accessible
echo -e "${YELLOW}🔍 Checking service health...${NC}"
if curl -f http://localhost:3000/health &> /dev/null; then
    echo -e "${GREEN}✅ Astra AI is running successfully!${NC}"
    echo -e "${GREEN}🌐 Access your app at: http://YOUR_DROPLET_IP:3000${NC}"
else
    echo -e "${YELLOW}⚠️  Service might still be starting. Check logs with: docker compose logs -f${NC}"
fi

echo -e "${GREEN}✅ Deployment complete!${NC}"
echo -e "${YELLOW}Useful commands:${NC}"
echo -e "  View logs: ${GREEN}docker compose logs -f${NC}"
echo -e "  Restart: ${GREEN}docker compose restart${NC}"
echo -e "  Stop: ${GREEN}docker compose down${NC}"
echo -e "  Rebuild: ${GREEN}docker compose up -d --build${NC}"
