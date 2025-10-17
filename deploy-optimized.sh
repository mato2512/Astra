#!/bin/bash
set -e

echo "=========================================="
echo "Astra AI - Optimized Deployment Script"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
REPO_DIR="/opt/astra"
SWAP_FILE="/swapfile"
SWAP_SIZE="4G"

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    print_error "Please run as root (use sudo)"
    exit 1
fi

# Step 1: Setup Swap Space
print_status "Checking swap space..."
if [ ! -f "$SWAP_FILE" ]; then
    print_status "Creating ${SWAP_SIZE} swap file..."
    fallocate -l $SWAP_SIZE $SWAP_FILE
    chmod 600 $SWAP_FILE
    mkswap $SWAP_FILE
    swapon $SWAP_FILE
    echo "$SWAP_FILE none swap sw 0 0" >> /etc/fstab
    print_status "Swap file created and activated"
else
    if ! swapon --show | grep -q $SWAP_FILE; then
        print_warning "Swap file exists but not active, activating..."
        swapon $SWAP_FILE
    fi
    print_status "Swap is active"
fi

free -h

# Step 2: Optimize System for Build
print_status "Optimizing system settings..."
sysctl -w vm.swappiness=10
sysctl -w vm.vfs_cache_pressure=50

# Step 3: Navigate to repository
print_status "Navigating to $REPO_DIR..."
cd $REPO_DIR

# Step 4: Pull latest changes
print_status "Pulling latest code from GitHub..."
git fetch origin
git reset --hard origin/main
git pull origin main

# Step 5: Clean Docker resources
print_status "Cleaning Docker resources..."
docker system prune -af --volumes || true

# Step 6: Stop existing containers
print_status "Stopping existing containers..."
docker compose -f docker-compose.prod.yaml down || true

# Step 7: Build with optimizations
print_status "Building Docker image (this will take 10-15 minutes)..."
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

# Build with progress output
docker compose -f docker-compose.prod.yaml build \
    --no-cache \
    --progress=plain

# Step 8: Start containers
print_status "Starting Astra AI..."
docker compose -f docker-compose.prod.yaml up -d

# Step 9: Wait for health check
print_status "Waiting for application to be healthy..."
sleep 10

# Check health for 2 minutes
for i in {1..24}; do
    if docker compose ps | grep -q "healthy"; then
        print_status "✓ Application is healthy!"
        break
    fi
    if [ $i -eq 24 ]; then
        print_warning "Health check timeout. Check logs with: docker compose logs"
        break
    fi
    echo "Waiting... ($i/24)"
    sleep 5
done

# Step 10: Display status
echo ""
print_status "=========================================="
print_status "Deployment Summary"
print_status "=========================================="
docker compose -f docker-compose.prod.yaml ps
echo ""
print_status "Memory Usage:"
free -h
echo ""
print_status "Container Logs (last 20 lines):"
docker compose -f docker-compose.prod.yaml logs --tail=20

echo ""
print_status "=========================================="
print_status "Deployment Complete!"
print_status "=========================================="
echo ""
print_status "Access your application:"
print_status "  Local: http://localhost:3000"
print_status "  Domain: http://your-domain.com"
echo ""
print_status "Useful commands:"
print_status "  View logs:    docker compose -f docker-compose.prod.yaml logs -f"
print_status "  Restart:      docker compose -f docker-compose.prod.yaml restart"
print_status "  Stop:         docker compose -f docker-compose.prod.yaml down"
print_status "  Shell access: docker exec -it astra-ai bash"
echo ""
