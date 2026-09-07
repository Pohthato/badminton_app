#!/bin/bash
# Badminton Analyzer Deployment Setup
# Run on Ubuntu 22.04+ server

set -e

APP_DIR="/var/www/badminton_app"
DOMAIN="your-domain.com"

echo "=== Badminton Analyzer Deployment Setup ==="

# 1. System dependencies
echo "[1/8] Installing system dependencies..."
sudo apt update
sudo apt install -y python3-pip python3-venv nginx redis-server supervisor git     libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev

# 2. Create app directory
echo "[2/8] Setting up app directory..."
sudo mkdir -p $APP_DIR
sudo chown -R $USER:$USER $APP_DIR

# 3. Clone/copy code
echo "[3/8] Copying application code..."
# If using git:
# git clone https://github.com/yourusername/badminton-analyzer.git $APP_DIR
# Otherwise, copy files manually to $APP_DIR

# 4. Python environment
echo "[4/8] Creating Python virtual environment..."
cd $APP_DIR
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 5. Create directories
echo "[5/8] Creating required directories..."
mkdir -p uploads static templates logs
mkdir -p /var/log/badminton_app

# 6. Environment file
echo "[6/8] Setting up environment..."
cat > .env << EOF
MOONSHOT_API_KEY=your_moonshot_api_key_here
REDIS_URL=redis://localhost:6379/0
SHUTTLE_MODEL_PATH=models/shuttlecock_yolov8n.pt
FLASK_ENV=production
EOF

echo "⚠️  IMPORTANT: Edit .env and add your real MOONSHOT_API_KEY"

# 7. Nginx setup
echo "[7/8] Configuring Nginx..."
sudo cp deploy/nginx.conf /etc/nginx/sites-available/badminton_app
sudo sed -i "s/your-domain.com/$DOMAIN/g" /etc/nginx/sites-available/badminton_app
sudo ln -sf /etc/nginx/sites-available/badminton_app /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx

# 8. Supervisor setup
echo "[8/8] Configuring Supervisor..."
sudo cp deploy/supervisor.conf /etc/supervisor/conf.d/badminton_app.conf
sudo supervisorctl reread
sudo supervisorctl update

echo ""
echo "=== Setup Complete ==="
echo "Next steps:"
echo "1. Edit $APP_DIR/.env with your API keys"
echo "2. Add your trained shuttlecock model to $APP_DIR/models/"
echo "3. Run: sudo supervisorctl start badminton_app badminton_worker badminton_beat"
echo "4. Set up SSL: sudo certbot --nginx -d $DOMAIN"
echo ""
echo "App will be available at: http://$DOMAIN"
