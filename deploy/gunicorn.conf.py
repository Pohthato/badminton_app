"""
Gunicorn configuration for production deployment.
Run with: gunicorn -c deploy/gunicorn.conf.py app_v2:app
"""
import multiprocessing
import os

bind = "0.0.0.0:5000"
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "sync"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
timeout = 120
keepalive = 5

# Logging
accesslog = "logs/gunicorn_access.log"
errorlog = "logs/gunicorn_error.log"
loglevel = "info"

# Process naming
proc_name = "badminton_analyzer"

# Server mechanics
daemon = False
pidfile = "logs/gunicorn.pid"

# SSL (configure if you have certificates)
# keyfile = "/path/to/key.pem"
# certfile = "/path/to/cert.pem"
