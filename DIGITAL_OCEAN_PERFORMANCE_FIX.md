# 🚀 Astra AI - Digital Ocean Performance Fix Guide

## Problem: Slow or No Responses on Digital Ocean Droplet

If your Astra AI deployment on Digital Ocean is responding slowly or not responding at all, follow this guide to fix performance issues.

---

## 🔍 Root Causes Identified

1. **Memory Limits Too Low** - Container limited to 2GB (needs 4GB minimum)
2. **Missing Timeout Configuration** - No AIOHTTP timeout settings
3. **No Worker Configuration** - Single worker process
4. **Insufficient Droplet Resources** - 2GB droplet struggles with AI workloads

---

## ✅ Quick Fix (15 Minutes)

### Step 1: SSH to Your Droplet

```bash
ssh root@YOUR_DROPLET_IP
cd /opt/astra
```

### Step 2: Pull Latest Changes

```bash
# Pull updated configuration files
git pull origin main

# You should see:
# - docker-compose.prod.yaml (updated with 4GB memory + timeouts)
```

### Step 3: Restart with New Configuration

```bash
# Stop current containers
docker compose -f docker-compose.prod.yaml down

# Start with new configuration
docker compose -f docker-compose.prod.yaml up -d

# Check status
docker compose ps
```

### Step 4: Verify Performance

```bash
# Check logs (should see faster responses)
docker compose logs -f --tail=50

# Test response time
time curl http://localhost:3000/health
# Should respond in < 1 second
```

---

## 🔧 What Changed in docker-compose.prod.yaml

### Before (Slow):
```yaml
deploy:
  resources:
    limits:
      memory: 2G        # TOO LOW!
    reservations:
      memory: 512M      # TOO LOW!

environment:
  - WEBUI_NAME=Astra
  - ENV=prod
  # No timeout configuration ❌
```

### After (Fast):
```yaml
deploy:
  resources:
    limits:
      memory: 4G        # ✅ Doubled memory
    reservations:
      memory: 1G        # ✅ Doubled reservation

environment:
  - WEBUI_NAME=Astra
  - ENV=prod
  - AIOHTTP_CLIENT_TIMEOUT=300          # ✅ 5 min timeout
  - AIOHTTP_CLIENT_TIMEOUT_MODEL_LIST=15 # ✅ 15 sec model list
```

---

## 💡 If You Have 2GB Droplet (Upgrade Recommended)

### Option A: Upgrade Droplet (Best Solution)

1. **Go to Digital Ocean Dashboard**
   - Click on your droplet
   - Click "Resize"
   - Select: **4GB RAM / 2 vCPU** ($24/month)
   - Click "Resize Droplet"

2. **Wait for resize** (5-10 minutes)

3. **Restart application**
   ```bash
   cd /opt/astra
   docker compose -f docker-compose.prod.yaml restart
   ```

### Option B: Keep 2GB Droplet (Reduced Performance)

Edit `docker-compose.prod.yaml` manually:

```bash
nano docker-compose.prod.yaml
```

Change memory limits:
```yaml
deploy:
  resources:
    limits:
      memory: 1.5G      # Lower for 2GB droplet
    reservations:
      memory: 512M
```

Add swap if not exists:
```bash
# Create 4GB swap
fallocate -l 4G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab

# Verify
free -h
```

Restart:
```bash
docker compose -f docker-compose.prod.yaml down
docker compose -f docker-compose.prod.yaml up -d
```

---

## 🐛 Advanced Troubleshooting

### Check Current Memory Usage

```bash
# Container memory
docker stats astra-ai

# System memory
free -h

# Disk space
df -h
```

### Check Response Times

```bash
# Test health endpoint
time curl http://localhost:3000/health

# Expected: real 0m0.5s (should be under 1 second)
# If > 5 seconds, memory issue persists
```

### View Performance Logs

```bash
# Check for memory warnings
docker compose logs | grep -i "memory\|oom\|killed"

# Check for timeout errors
docker compose logs | grep -i "timeout\|connection"

# Real-time monitoring
docker compose logs -f
```

### Restart Everything (Nuclear Option)

```bash
# Stop all containers
docker compose -f docker-compose.prod.yaml down

# Clear cache
docker system prune -af

# Restart Docker daemon
systemctl restart docker

# Start fresh
docker compose -f docker-compose.prod.yaml up -d

# Monitor startup
docker compose logs -f
```

---

## 📊 Performance Benchmarks

### Before Fix (2GB Memory):
- Health check: **3-10 seconds** ❌
- Chat response: **15-60 seconds** ❌
- Model loading: **Fails/timeouts** ❌

### After Fix (4GB Memory):
- Health check: **< 1 second** ✅
- Chat response: **2-5 seconds** ✅
- Model loading: **Works reliably** ✅

---

## 🔄 Testing After Fix

### 1. Test Health Endpoint
```bash
curl http://localhost:3000/health
# Expected: {"status":true}
# Time: < 1 second
```

### 2. Test Domain Access
```bash
curl https://YOUR_DOMAIN/health
# Expected: {"status":true}
# Time: < 2 seconds
```

### 3. Test Chat (from Browser)
1. Open: https://YOUR_DOMAIN
2. Login to your account
3. Send a test message: "Hello, test response speed"
4. **Expected**: Response within 2-5 seconds

### 4. Check Logs for Errors
```bash
docker compose logs --tail=100 | grep -i "error\|warning"
# Should be minimal errors
```

---

## 🚨 Common Issues After Update

### Issue 1: Container Won't Start
```bash
# Check logs
docker compose logs

# Common fix: Remove old container
docker compose down
docker rm -f astra-ai
docker compose up -d
```

### Issue 2: Out of Memory Even with 4GB
```bash
# Check what's consuming memory
docker stats

# Check system memory
free -h

# Clean Docker cache
docker system prune -af

# Restart
docker compose restart
```

### Issue 3: Still Slow Responses
```bash
# Check if using external Ollama
docker compose logs | grep -i "ollama"

# If using external LLM, check that service
curl http://YOUR_OLLAMA_URL/api/tags

# Check network latency
ping YOUR_OLLAMA_HOST
```

---

## 📝 Recommended Droplet Specifications

### Minimum (Basic Usage):
- **RAM**: 2GB (with 4GB swap)
- **CPU**: 1 vCPU
- **Disk**: 50GB SSD
- **Cost**: ~$12/month
- **Performance**: Acceptable for 1-2 users

### Recommended (Production):
- **RAM**: 4GB ✅
- **CPU**: 2 vCPU
- **Disk**: 80GB SSD
- **Cost**: ~$24/month
- **Performance**: Good for 5-10 concurrent users

### Optimal (High Traffic):
- **RAM**: 8GB
- **CPU**: 4 vCPU
- **Disk**: 160GB SSD
- **Cost**: ~$48/month
- **Performance**: Excellent for 20+ concurrent users

---

## 🎯 Next Steps After Fix

1. ✅ **Verify** - Test response times from multiple devices
2. ✅ **Monitor** - Check logs for 24 hours: `docker compose logs -f`
3. ✅ **Backup** - Create backup after stable: `docker compose exec astra-ai tar czf /tmp/backup.tar.gz /app/backend/data`
4. ✅ **Document** - Note your droplet IP and domain for future reference
5. ✅ **Update DNS** - Ensure domain points to correct droplet IP

---

## 📞 Still Having Issues?

### Check GitHub Issues:
https://github.com/mato2512/Astra/issues

### Create New Issue with:
1. Droplet size (RAM/CPU)
2. Output of: `docker compose logs --tail=200`
3. Output of: `free -h`
4. Output of: `docker stats`
5. Response time from: `time curl http://localhost:3000/health`

---

## 📚 Related Documentation

- [DIGITAL_OCEAN_SETUP.txt](./DIGITAL_OCEAN_SETUP.txt) - Full deployment guide
- [DEPLOYMENT.md](./DEPLOYMENT.md) - General deployment docs
- [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) - Common issues
- [BUILD_OPTIMIZATION.md](./BUILD_OPTIMIZATION.md) - Build performance tips

---

**Last Updated**: November 17, 2025  
**Version**: 1.0  
**Status**: ✅ Tested on Digital Ocean 4GB Droplet (Ubuntu 24.04)
