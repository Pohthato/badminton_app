# GPU Testing Platforms for OmniCourt

## 🎯 Quick Comparison

| Platform | GPU | Free Tier | Setup | Best For |
|----------|-----|-----------|-------|----------|
| **Kaggle** | T4/P100 | ✅ 40 hrs/week | Easy | Quick testing |
| **Google Colab** | T4/L4/V100 | ✅ 12 hrs/session | Very Easy | Development |
| **AWS SageMaker** | Various | ⚠️ Limited trial | Medium | Production |
| **Paperspace** | V100/A40 | ⚠️ $10/month | Medium | Long-term testing |
| **Lambda Labs** | A40/A6000 | ⚠️ $0.51/hr | Easy | High performance |
| **Modal.com** | A10/H100 | ✅ Limited free | Very Easy | Serverless |
| **HuggingFace Spaces** | T4 | ✅ Free | Very Easy | Demo/Streamlit |
| **Replicate.com** | Various | ✅ Credits | Medium | API-based |

---

## 1️⃣ **Google Colab** (Recommended for Development) ⭐⭐⭐⭐⭐

### Pros:
- ✅ **Free** (12 hours per session, up to 40 hrs/week)
- ✅ Very easy setup
- ✅ Multiple GPU options (T4, L4, V100)
- ✅ Full Jupyter notebook support
- ✅ Pre-installed common libraries

### Setup:
```python
# First cell
!git clone <your-repo> omnicourt
%cd omnicourt

# Install dependencies
!pip install -q -r requirements_professional.txt

# Download models
!python -c "from ultralytics import YOLO; YOLO('yolov8s-pose.pt')"

# Start Flask in background
import subprocess
subprocess.Popen(['python', 'app_professional.py'])

# Test health
import urllib.request
import time
time.sleep(3)
response = urllib.request.urlopen('http://localhost:5000/health')
print(response.read().decode())
```

### Cons:
- ⚠️ 12-hour session timeout
- ⚠️ RAM limited to 13GB
- ⚠️ Can't use as persistent server

---

## 2️⃣ **Kaggle** (Current Choice) ⭐⭐⭐⭐

### Pros:
- ✅ 40 hours/week free GPU
- ✅ Pre-loaded with common libraries
- ✅ Can upload large datasets
- ✅ T4 GPU

### Cons:
- ⚠️ Limited to notebooks
- ⚠️ Can't easily tunnel to public internet
- ⚠️ No persistent processes

### Fixed Setup for Kaggle:
```python
# Cell 1: System setup
!apt-get update -qq && apt-get install -y redis-server ffmpeg > /dev/null 2>&1

# Cell 2: Install Python packages
!pip install -q -r /kaggle/input/datasets/pohthato/badminton-app/omnicourt_v5_flowfix/requirements_professional.txt

# Cell 3: Start services
import subprocess
import time
import urllib.request

print("Starting Redis...")
redis_proc = subprocess.Popen(['redis-server', '--port', '6379'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
time.sleep(2)

print("Starting Flask...")
os.chdir('/kaggle/input/datasets/pohthato/badminton-app/omnicourt_v5_flowfix')
flask_proc = subprocess.Popen(['python', 'app_professional.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
time.sleep(5)

# Test health
try:
    response = urllib.request.urlopen('http://localhost:5000/health', timeout=5)
    print(f"✅ Success: {response.read().decode()}")
except Exception as e:
    print(f"❌ Error: {e}")

# Cell 4: Use the API
# Upload a video and test
import requests
with open('/path/to/video.mp4', 'rb') as f:
    files = {'video': f}
    response = requests.post('http://localhost:5000/detect-players', files=files)
    print(response.json())
```

---

## 3️⃣ **Paperspace** (Recommended for Longer Testing) ⭐⭐⭐⭐

### Pros:
- ✅ More GPU hours than Colab
- ✅ Persistent instances
- ✅ V100 GPUs available
- ✅ Can tunnel to public internet
- ✅ Jupyter notebooks

### Cons:
- ⚠️ $10/month minimum for persistent machine
- ⚠️ Pay-per-hour otherwise

### Setup:
```bash
# Create free account at paperspace.com

# In terminal:
git clone <repo> omnicourt
cd omnicourt

pip install -r requirements_professional.txt
python -m pip install cloudflared  # For tunneling

# Start Flask
python app_professional.py

# In another terminal, tunnel it
cloudflared tunnel --url http://localhost:5000
```

---

## 4️⃣ **Lambda Labs** (Best GPU Performance) ⭐⭐⭐⭐

### Pros:
- ✅ High-end GPUs (A6000, A100)
- ✅ $0.51/hr for A40 (very cheap)
- ✅ Can SSH in directly
- ✅ Jupyter support

### Cons:
- ⚠️ Pay-per-hour (no free tier)
- ⚠️ Must reserve instances

### Setup:
```bash
# 1. Sign up at lambdalabs.com
# 2. Launch an A40 instance (~$0.51/hr)
# 3. SSH into the instance

git clone <repo> omnicourt
cd omnicourt

pip install -r requirements_professional.txt

# Start Flask with ngrok tunnel
pip install ngrok
python app_professional.py &

# In another terminal
pip install pyngrok
python -c "
from pyngrok import ngrok
public_url = ngrok.connect(5000)
print(f'Public URL: {public_url}')
" &

# Keep terminal open or use screen/tmux
```

---

## 5️⃣ **AWS SageMaker** (Enterprise Grade) ⭐⭐⭐

### Pros:
- ✅ Multiple GPU options
- ✅ Production-ready
- ✅ Auto-scaling capability
- ✅ Integration with AWS services

### Cons:
- ⚠️ Steep learning curve
- ⚠️ Complex pricing
- ⚠️ Overkill for testing

---

## 6️⃣ **Modal.com** (Serverless, New!) ⭐⭐⭐⭐

### Pros:
- ✅ Free tier available
- ✅ Easy Python API
- ✅ A10/H100 GPUs
- ✅ Serverless (no setup)

### Cons:
- ⚠️ Newer platform
- ⚠️ Limited free credits

### Setup:
```bash
pip install modal

# Create app.py
import modal

app = modal.App("omnicourt")

@app.function(gpu="T4")
def analyze_video(video_path):
    # Your analysis code
    return results

if __name__ == "__main__":
    result = analyze_video.remote("video.mp4")
    print(result)
```

---

## 7️⃣ **Replicate.com** (API-Based Testing) ⭐⭐⭐

### Pros:
- ✅ Easy API deployment
- ✅ No infrastructure setup
- ✅ Pay per inference
- ✅ Good for production APIs

### Cons:
- ⚠️ Limited customization
- ⚠️ Pay-per-request pricing

---

## 🎯 Recommended Path for OmniCourt

### Phase 1: Quick Testing
→ **Use Google Colab** (free, easiest)

```python
# Colab cell
!git clone https://github.com/your-repo omnicourt
%cd omnicourt
!pip install -q -r requirements_professional.txt
!python app_professional.py &
import time; time.sleep(5)

# Test with sample video
import requests
with open('sample_video.mp4', 'rb') as f:
    r = requests.post('http://localhost:5000/detect-players', files={'video': f})
    print(r.json())
```

### Phase 2: Extended Testing
→ **Use Kaggle** (40 hrs/week free)

```python
# Kaggle notebook
# Follow the fixed Kaggle setup above
```

### Phase 3: Production Testing
→ **Use Paperspace** ($10/month) or **Lambda Labs** (~$0.51/hr for A40)

```bash
# SSH into instance
git clone <repo>
cd repo
pip install -r requirements_professional.txt
python app_professional.py &

# Tunnel it
pip install cloudflared
cloudflared tunnel --url http://localhost:5000
```

---

## 🔧 Fix for Your Current Kaggle Error

**Problem**: `FileNotFoundError: [Errno 2] No such file or directory: 'celery'`

**Solution**: Use Python module syntax instead

```python
# ❌ Wrong (doesn't work on Kaggle)
subprocess.Popen(['celery', '-A', 'tasks', 'worker'])

# ✅ Correct
subprocess.Popen([sys.executable, '-m', 'celery', '-A', 'tasks', 'worker'])
```

Or better yet, **skip Celery on Kaggle** since it's not needed for synchronous testing:

```python
# Modified for Kaggle (no Celery)
import subprocess
import time
import os

# Just Redis + Flask
subprocess.Popen(['redis-server', '--port', '6379'])
time.sleep(2)
print("✅ Redis started")

subprocess.Popen(['python', 'app_professional.py'])
time.sleep(5)
print("✅ Flask started on port 5000")

# Test health
import urllib.request
try:
    response = urllib.request.urlopen('http://localhost:5000/health', timeout=5)
    print(f"✅ {response.read().decode()}")
except Exception as e:
    print(f"❌ {e}")
```

---

## 💡 My Recommendation for You

**For testing OmniCourt on GPU:**

1. **First**: Try **Google Colab** for quick validation (1-2 hours)
2. **Then**: Move to **Kaggle** for longer testing (using fixed script)
3. **Finally**: Deploy on **Lambda Labs** for production testing ($0.51/hr for A40 is very cheap)

**Why this order?**
- Colab: Fastest iteration, no setup
- Kaggle: More GPU hours, familiar environment
- Lambda: Real-world testing with persistent server

---

## 📋 Testing Checklist for Any Platform

```python
# Test each component
1. ✅ Health check: curl http://localhost:5000/health
2. ✅ Upload video: POST /detect-players
3. ✅ Court calibration: POST /calibrate-court
4. ✅ Live analysis: GET /analyze-live (SSE stream)
5. ✅ Player report: GET /player-report/0
6. ✅ Analytics dashboard: Load HTML
7. ✅ Real-time metrics: Check FPS and latency
```

---

Need help setting up any of these? Let me know which platform you'd like to use!
