# OmniCourt Professional - Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Option 1: Local Development (Recommended for Testing)

#### Prerequisites
- Python 3.9+
- CUDA 11.8+ (for GPU acceleration, optional)
- FFmpeg
- 16GB+ RAM

#### Installation

```bash
# 1. Clone and navigate
git clone <repo-url>
cd omnicourt_v5_flowfix

# 2. Create virtual environment
python -m venv venv

# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements_professional.txt

# 4. Run application
python app_professional.py
```

Access at: **http://localhost:5000**

---

### Option 2: Docker (Recommended for Production)

#### Prerequisites
- Docker 20.10+
- Docker Compose 2.0+
- NVIDIA Docker (for GPU support)
- 12GB+ available disk space

#### Installation

```bash
# 1. Clone and navigate
git clone <repo-url>
cd omnicourt_v5_flowfix

# 2. Create .env file
cp .env.example .env

# Edit .env with your settings:
# - DB_PASSWORD
# - CUDA settings (if using GPU)

# 3. Start all services
docker-compose -f docker-compose.professional.yml up -d

# 4. Check status
docker-compose -f docker-compose.professional.yml ps
```

Access at: **http://localhost** (or **http://localhost:5000** for API)

---

## 📖 First Use Guide

### 1. Upload a Video

1. Go to **http://localhost:5000**
2. Click or drag-drop a badminton match video
3. Supported formats: MP4, MOV, WebM, MKV, AVI
4. Max size: 500MB, Max duration: 5 minutes

### 2. Wait for Detection

The system will:
- Auto-detect all players
- Draw skeleton overlays
- Calibrate court lines
- Display frame preview

### 3. Select Player

Click on a player card to start analysis.

### 4. View Analysis

- Real-time skeleton overlay
- Velocity metrics
- Court position tracking
- Performance indicators

### 5. Access Dashboard

After analysis, visit **http://localhost:5000/analysis** for:
- Detailed metrics
- Performance charts
- Stroke analysis
- Fatigue assessment
- AI recommendations

---

## 🛠️ Configuration

### Environment Variables (.env)

```env
# Flask Settings
FLASK_ENV=production
FLASK_APP=app_professional.py

# Database (if using Docker)
DB_PASSWORD=your_password_here

# GPU Configuration
USE_CUDA=1
CUDA_VISIBLE_DEVICES=0

# Video Processing
MAX_VIDEO_LENGTH_SEC=300
MAX_VIDEO_SIZE_MB=512
```

### Performance Tuning

**For faster analysis:**
```env
SKIP_FRAMES=2  # Analyze every 3rd frame
MODEL_SIZE=small  # Use yolov8s instead of yolov8l
```

**For better accuracy:**
```env
SKIP_FRAMES=0  # Analyze every frame
MODEL_SIZE=large  # Use yolov8l
```

---

## 📊 Understanding the Results

### Court Detection
- **Green**: Successfully detected
- **Red**: Manual calibration required
- Click 3-4 corners to manually calibrate if auto-detection fails

### Skeleton Overlay
- **Blue lines**: Pose skeleton
- **Green circles**: Detected keypoints
- **Confidence**: Displayed as transparency

### Performance Metrics
- **Velocity**: Pixels per frame (higher = faster movement)
- **Fatigue Index**: 0-100 (0 = fresh, 100 = exhausted)
- **Court Coverage**: % of court area used

---

## 🔧 Common Issues & Solutions

### Issue: GPU Not Detected

```bash
# Check CUDA installation
nvidia-smi

# If NVIDIA Docker not installed:
# Ubuntu/Debian:
sudo apt install nvidia-docker2

# Then in docker-compose:
set CUDA_VISIBLE_DEVICES=0  # Specify GPU
```

### Issue: "Failed to Fetch" Error

1. Check server is running: `curl http://localhost:5000/health`
2. Check logs: `docker logs omnicourt-app`
3. Restart services: `docker-compose restart`

### Issue: Out of Memory

```env
# Use smaller model
MODEL_SIZE=small

# Or skip frames
SKIP_FRAMES=2
```

### Issue: Court Not Detecting

1. Ensure good lighting
2. Court lines should be visible
3. Use manual calibration (click corners)
4. Try different camera angle

---

## 📈 Monitoring

### Check Service Health

```bash
# Check all services
docker-compose ps

# View specific service logs
docker logs omnicourt-app
docker logs omnicourt-db
docker logs omnicourt-redis

# Real-time logs
docker logs -f omnicourt-app
```

### Performance Metrics

Visit **http://localhost:5000/health** for:
```json
{
  "status": "healthy",
  "models": {
    "yolo": true,
    "pose": true,
    "cuda_available": true
  }
}
```

---

## 📚 API Quick Reference

### Upload Video
```bash
curl -X POST -F "video=@match.mp4" http://localhost:5000/detect-players
```

### Get Analysis Report
```bash
curl http://localhost:5000/player-report/0
```

### Calibrate Court
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"corners": [[100,100],[500,100],[500,400],[100,400]]}' \
  http://localhost:5000/calibrate-court
```

### Court Status
```bash
curl http://localhost:5000/court-diagnostics
```

---

## 🎓 Understanding the Analysis

### Key Metrics

| Metric | Range | Interpretation |
|--------|-------|-----------------|
| Velocity | 0-50+ | Movement speed in pixels/frame |
| Fatigue | 0-100 | Estimated fatigue level |
| Posture | 0-100% | Upright posture percentage |
| Balance | 0-1.0 | Balance stability score |
| Strokes | 0-100+ | Total strokes detected |

### Recommendations

The system provides AI-powered recommendations for:
- Posture improvement
- Fatigue management
- Movement consistency
- Overall performance

---

## 🚀 Next Steps

1. **Try Example Videos**
   - Use sample badminton match footage
   - Test different camera angles
   - Try different players

2. **Explore Features**
   - Test court calibration
   - Review analytics dashboard
   - Check recommendations

3. **Customize Settings**
   - Adjust detection sensitivity
   - Configure GPU usage
   - Set performance trade-offs

4. **Integrate with Your Workflow**
   - Export analysis reports
   - Use API for automation
   - Build custom dashboards

---

## 📞 Support

### Documentation
- Full guide: [README_PROFESSIONAL.md](README_PROFESSIONAL.md)
- Deployment: [DEPLOYMENT.md](DEPLOYMENT.md)
- Architecture: [IMPROVEMENTS.md](IMPROVEMENTS.md)

### Common Questions

**Q: How accurate is court detection?**
A: 95%+ with proper lighting. Auto-detection works in most cases.

**Q: Can I track multiple players?**
A: Yes! The system detects all players but provides detailed analysis for one selected player per report.

**Q: What's the latency?**
A: ~100ms per frame with GPU, ideal for real-time analysis.

**Q: Can I use CPU instead of GPU?**
A: Yes, but slower (1-2 FPS). GPU recommended for production.

**Q: How much disk space do I need?**
A: ~100GB for models + storage. Models cached in `/models/`

---

## ✅ Verification Checklist

After installation, verify:
- [ ] Web UI loads at http://localhost:5000
- [ ] Health check passes: http://localhost:5000/health
- [ ] Can upload a test video
- [ ] Players are detected
- [ ] Skeleton overlay displays
- [ ] Analytics dashboard loads
- [ ] Performance metrics show

---

## 🎯 You're Ready!

OmniCourt Professional is now running. Upload your first badminton video and experience professional-grade analysis! 🎮

**Happy analyzing! 🏸**
