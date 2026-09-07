"""
OmniCourt Kaggle GPU Test Startup Script
Starts Flask + Redis locally (Celery skipped on Kaggle)
"""
import subprocess
import time
import urllib.request
import re
import os
import sys

print("🚀 Starting OmniCourt on Kaggle GPU...")

# Check if running on Kaggle
IS_KAGGLE = os.path.exists('/kaggle')

# 1. Start Redis (if available)
print("\n📦 Starting Redis...")
try:
    # Kaggle has Redis, but use foreground mode
    redis_proc = subprocess.Popen(
        ['redis-server', '--port', '6379'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    time.sleep(2)
    print("✅ Redis started on port 6379")
except FileNotFoundError:
    print("⚠️  Redis not found, skipping...")
    redis_proc = None

# 2. Start Flask App
print("\n🌐 Starting Flask App...")
flask_proc = subprocess.Popen(
    ['python', 'app_professional.py'],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)
time.sleep(5)

# Check if Flask started successfully
if flask_proc.poll() is None:
    print("✅ Flask App started on port 5000")
else:
    print("❌ Flask failed to start. Checking logs...")
    stdout, stderr = flask_proc.communicate()
    print("STDOUT:", stdout)
    print("STDERR:", stderr)
    sys.exit(1)

# 3. Test the server
print("\n🧪 Testing Flask health endpoint...")
try:
    response = urllib.request.urlopen('http://localhost:5000/health', timeout=5)
    data = response.read().decode()
    print(f"✅ Health check passed: {data}")
except Exception as e:
    print(f"⚠️  Health check failed: {e}")

# 4. Create public tunnel (Kaggle doesn't support cloudflared)
if IS_KAGGLE:
    print("\n⚠️  Note: Cloudflared tunneling not available on Kaggle")
    print("   Use Kaggle's public dataset sharing or download results instead")
else:
    # If not on Kaggle, try cloudflared
    print("\n🌐 Creating secure public tunnel with Cloudflare...")
    try:
        cloudflared_proc = subprocess.Popen(
            ['cloudflared', 'tunnel', '--url', 'http://localhost:5000'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # Capture the public URL
        for line in cloudflared_proc.stdout:
            match = re.search(r'https://[a-zA-Z0-9\-]+\.trycloudflare\.com', line)
            if match:
                url = match.group(0)
                print("\n" + "="*60)
                print(f"🎉 SUCCESS! YOUR PUBLIC URL IS:")
                print(f"👉 {url}")
                print("="*60 + "\n")
                break
    except FileNotFoundError:
        print("⚠️  cloudflared not installed")

print("\n" + "="*60)
print("✅ OmniCourt is running!")
print("   Local API: http://localhost:5000")
print("   Health check: http://localhost:5000/health")
print("="*60)

# Keep processes running
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n🛑 Shutting down...")
    flask_proc.terminate()
    if redis_proc:
        redis_proc.terminate()
