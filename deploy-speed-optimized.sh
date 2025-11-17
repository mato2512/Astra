#!/bin/bash
# Quick deployment script for speed optimizations
# Run on your Digital Ocean droplet

echo "🚀 Deploying Astra AI Speed Optimizations..."

# Stop current container
echo "📦 Stopping current container..."
docker-compose -f docker-compose.prod.yaml down

# Pull latest code
echo "⬇️  Pulling latest changes..."
git pull origin main

# Rebuild with no cache to ensure all changes applied
echo "🔨 Rebuilding image (this takes 5-10 minutes)..."
docker-compose -f docker-compose.prod.yaml build --no-cache

# Start with optimizations
echo "▶️  Starting optimized container..."
docker-compose -f docker-compose.prod.yaml up -d

# Wait for startup
echo "⏳ Waiting for health check..."
sleep 10

# Show status
echo ""
echo "📊 Container Status:"
docker ps | grep astra-ai

echo ""
echo "💾 Resource Usage:"
docker stats --no-stream astra-ai

echo ""
echo "📝 Recent Logs:"
docker logs --tail 30 astra-ai

echo ""
echo "✅ Deployment complete!"
echo ""
echo "🧪 Test your chat now - should respond in <1 second!"
echo "📈 Monitor with: docker stats astra-ai"
echo "📋 View logs with: docker logs -f astra-ai"
