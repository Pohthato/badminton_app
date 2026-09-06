# OmniCourt - Badminton AI Analyzer v3

## Features
- **Player Detection**: YOLOv8 custom badminton action model
- **Skeleton Tracking**: YOLOv8-Pose (40-60 FPS on GPU)
- **Court Mapping**: Auto + manual homography for real-world metrics
- **AI Coaching**: DeepSeek API for personalized feedback
- **Background Processing**: Celery + Redis for long videos

## Quick Start (Local)
```bash
pip install -r requirements_v2.txt
# Start Redis
docker run -d -p 6379:6379 redis:7-alpine
# Start Celery worker
celery -A tasks worker --loglevel=info --pool=solo
# Start Flask
python app_v2.py
```

## Quick Start (Docker)
```bash
docker-compose up -d
```

## Kaggle Deployment
1. Upload all files to a Kaggle Dataset
2. Create notebook with the provided cells
3. Run all cells → get Cloudflared URL

## Environment Variables
Copy `.env.example` to `.env` and set:
- `DEEPSEEK_API_KEY` - Your DeepSeek API key
- `REDIS_URL` - Redis connection string

## Video Limits
- Max duration: 2 minutes (configurable in tasks.py)
- Max file size: 512MB
- Max frames: 3000 (~2 min at 25fps)

## Court Calibration
- Auto-detects court lines on first frame
- Manual override: click "Mark Court Corners" on upload page
- Enter 4 corner coordinates (TL, TR, BR, BL)

## Fixed Bugs (v3)
- Python 3.12 f-string syntax errors
- NumPy 2.x dtype casting crashes
- Slow keypointrcnn replaced with YOLOv8-Pose (10x faster)
- Safe Celery task status handling
- Array shape validation in pose processing
