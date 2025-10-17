# Digital Ocean Docker Deployment Troubleshooting Guide

## Common Errors and Solutions

### Error 1: "JavaScript heap out of memory"
**Cause:** Not enough RAM during Node.js build

**Solution:**
```bash
# Add swap space on your droplet
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Then rebuild
docker compose build --no-cache
```

### Error 2: "npm ci --legacy-peer-deps failed"
**Cause:** Package dependency conflicts

**Solution:**
Already fixed in Dockerfile. Ensure you have the latest code:
```bash
git pull origin main
```

### Error 3: "y-protocols/awareness not found"
**Cause:** Missing external module configuration

**Solution:**
Already fixed in `vite.config.ts`. Pull latest code:
```bash
git pull origin main
```

### Error 4: Docker build takes too long or hangs
**Cause:** Large build context, slow network

**Solution:**
```bash
# Use BuildKit for better caching
export DOCKER_BUILDKIT=1

# Build with progress
docker compose build --progress=plain

# Or build individual stage
docker build --target=build -t astra-build .
```

### Error 5: "Cannot connect to Docker daemon"
**Cause:** Docker service not running or permission issues

**Solution:**
```bash
# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Verify
docker ps
```

### Error 6: Port already in use
**Cause:** Another service using port 3000

**Solution:**
```bash
# Check what's using port 3000
sudo lsof -i :3000

# Kill the process or use different port
# Edit .env file:
PORT=8080

# Then restart
docker compose down
docker compose up -d
```

### Error 7: "Container exits immediately"
**Cause:** Backend startup failure

**Solution:**
```bash
# Check logs
docker compose logs astra-ai

# Common causes:
# - Missing environment variables
# - Database permission issues
# - Port conflicts

# Try running interactively
docker compose run --rm astra-ai bash
```

### Error 8: Build fails with "no space left on device"
**Cause:** Disk space full

**Solution:**
```bash
# Check disk space
df -h

# Clean up Docker
docker system prune -a --volumes

# Remove old images
docker image prune -a

# Clean package cache
sudo apt clean
```

## Deployment Steps (Digital Ocean)

### 1. Initial Setup
```bash
# SSH into your droplet
ssh root@YOUR_DROPLET_IP

# Update system
apt update && apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose
apt install docker-compose-plugin -y

# Verify installation
docker --version
docker compose version
```

### 2. Clone Repository
```bash
# Create app directory
mkdir -p /opt/astra
cd /opt/astra

# Clone repo
git clone https://github.com/mato2512/Astra.git .

# Create environment file
cp .env.example .env
nano .env  # Edit your settings
```

### 3. Add Swap Space (IMPORTANT for small droplets)
```bash
# Create 4GB swap
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### 4. Build and Deploy
```bash
# Method 1: Using deployment script
chmod +x deploy-digital-ocean.sh
./deploy-digital-ocean.sh

# Method 2: Manual
export DOCKER_BUILDKIT=1
docker compose -f docker-compose.prod.yaml build
docker compose -f docker-compose.prod.yaml up -d

# Check status
docker compose ps
docker compose logs -f
```

### 5. Configure Firewall
```bash
# Allow HTTP traffic
ufw allow 3000/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw enable
```

### 6. Set Up Reverse Proxy (Optional but recommended)

#### Using Nginx:
```bash
apt install nginx -y

# Create nginx config
cat > /etc/nginx/sites-available/astra <<EOF
server {
    listen 80;
    server_name YOUR_DOMAIN_OR_IP;

    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_cache_bypass \$http_upgrade;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

# Enable site
ln -s /etc/nginx/sites-available/astra /etc/nginx/sites-enabled/
nginx -t
systemctl reload nginx
```

## Monitoring & Maintenance

### View Logs
```bash
# All logs
docker compose logs -f

# Specific service
docker compose logs -f astra-ai

# Last 100 lines
docker compose logs --tail=100
```

### Restart Services
```bash
# Restart all
docker compose restart

# Restart specific service
docker compose restart astra-ai
```

### Update Astra
```bash
cd /opt/astra
git pull origin main
docker compose down
docker compose build --no-cache
docker compose up -d
```

### Backup Data
```bash
# Backup volume
docker run --rm -v astra-data:/data -v $(pwd):/backup ubuntu tar czf /backup/astra-backup-$(date +%Y%m%d).tar.gz /data

# Restore volume
docker run --rm -v astra-data:/data -v $(pwd):/backup ubuntu tar xzf /backup/astra-backup-YYYYMMDD.tar.gz -C /
```

## Performance Optimization

### For Small Droplets (1GB RAM)
```bash
# Limit container memory
# Edit docker-compose.prod.yaml:
deploy:
  resources:
    limits:
      memory: 768M
```

### For Build Performance
```bash
# Use BuildKit cache
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

# Build with cache
docker compose build --build-arg BUILDKIT_INLINE_CACHE=1
```

## Getting Help

If issues persist:

1. **Check logs**: `docker compose logs -f`
2. **Check container status**: `docker compose ps`
3. **Check system resources**: `htop` or `free -h`
4. **Test connectivity**: `curl http://localhost:3000/health`
5. **Verify environment**: `docker compose config`

## Minimum Recommended Specifications

- **Droplet Size**: 2GB RAM minimum (4GB recommended)
- **Storage**: 20GB minimum
- **OS**: Ubuntu 22.04 LTS or newer
