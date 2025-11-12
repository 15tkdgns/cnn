# Food-101 Classifier - Deployment Guide

Complete deployment guide for the Food-101 image classification application.

## System Overview

This application consists of three main components:

1. **Model Training** (`notebooks/`) - Jupyter notebook for training ResNet18 on Food-101
2. **Backend API** (`api/`) - FastAPI server for model inference
3. **Frontend** (`frontend/`) - React web application with ChatGPT-style UI

## Prerequisites

- Python 3.8+
- Node.js 14+
- npm or yarn
- CUDA-compatible GPU (optional, for faster inference)
- Trained model file: `outputs/models/best_model.pth`

## Quick Start

### 1. Start the Backend API

```bash
# Navigate to API directory
cd api

# Install Python dependencies
pip install -r requirements.txt

# Start the server
python main.py

# Or use uvicorn with auto-reload
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at: `http://localhost:8000`

API Endpoints:
- `GET /` - Web interface (or API info)
- `GET /health` - Health check
- `GET /classes` - List of 101 food classes
- `POST /predict` - Single image prediction
- `POST /predict/batch` - Batch prediction (up to 10 images)
- `GET /docs` - Swagger UI documentation

### 2. Start the Frontend

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies (first time only)
npm install

# Start development server
npm start
```

The React app will open at: `http://localhost:3000`

### 3. Use the Application

1. Open `http://localhost:3000` in your browser
2. Upload a food image by:
   - Dragging and dropping
   - Clicking the upload zone
   - Pasting with Ctrl+V
3. Click "분석하기" (Analyze) to classify
4. View the predicted food class and Top-5 results

## Production Deployment

### Backend (FastAPI)

#### Option 1: Docker

```bash
cd api

# Create Dockerfile
cat > Dockerfile << EOF
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

# Build and run
docker build -t food101-api .
docker run -p 8000:8000 food101-api
```

#### Option 2: Systemd Service

```bash
# Create service file
sudo nano /etc/systemd/system/food101-api.service

# Add content:
[Unit]
Description=Food-101 Classification API
After=network.target

[Service]
User=your-user
WorkingDirectory=/path/to/llm_prj/api
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target

# Enable and start
sudo systemctl enable food101-api
sudo systemctl start food101-api
```

#### Option 3: Gunicorn + Nginx

```bash
# Install gunicorn
pip install gunicorn

# Run with gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Nginx config
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Frontend (React)

#### Option 1: Build and Serve Static Files

```bash
cd frontend

# Build production files
npm run build

# Serve with simple HTTP server
npx serve -s build -p 3000

# Or copy to nginx
sudo cp -r build/* /var/www/html/
```

#### Option 2: Nginx Configuration

```nginx
server {
    listen 80;
    server_name your-domain.com;

    root /var/www/food101-frontend/build;
    index index.html;

    location / {
        try_files $uri /index.html;
    }

    # Proxy API requests to backend
    location /api/ {
        proxy_pass http://localhost:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

#### Option 3: Docker

```bash
cd frontend

# Create Dockerfile
cat > Dockerfile << EOF
FROM node:18-alpine as build

WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/build /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
EOF

# Build and run
docker build -t food101-frontend .
docker run -p 3000:80 food101-frontend
```

## Environment Configuration

### Backend (.env)

```env
# API Configuration
HOST=0.0.0.0
PORT=8000
WORKERS=4

# Model Configuration
MODEL_PATH=../outputs/models/best_model.pth
DEVICE=cuda  # or cpu

# Data Configuration
DATA_PATH=../data/food-101

# CORS
CORS_ORIGINS=http://localhost:3000,https://your-domain.com
```

### Frontend (.env)

```env
# API URL
REACT_APP_API_URL=http://localhost:8000

# Production
REACT_APP_API_URL=https://api.your-domain.com
```

## Performance Optimization

### Backend

1. **Use GPU**: Ensure CUDA is available for faster inference
2. **Batch Processing**: Use `/predict/batch` for multiple images
3. **Caching**: Add Redis for caching predictions
4. **Load Balancing**: Use multiple workers with Gunicorn
5. **Model Optimization**: Convert to TorchScript or ONNX

### Frontend

1. **Code Splitting**: Already configured with React
2. **Image Optimization**: Compress images before upload
3. **CDN**: Serve static files from CDN
4. **Lazy Loading**: Implement lazy loading for components

## Monitoring

### Backend Health Check

```bash
# Check API health
curl http://localhost:8000/health

# Expected response:
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "num_classes": 101
}
```

### Logs

```bash
# View backend logs
tail -f /var/log/food101-api.log

# Or with systemd
journalctl -u food101-api -f
```

## Troubleshooting

### Backend Issues

**Model not found:**
```bash
# Ensure model file exists
ls outputs/models/best_model.pth

# If missing, train the model first
cd notebooks
jupyter notebook food101_training_optimal.ipynb
```

**CUDA out of memory:**
- Reduce batch size in batch prediction
- Use CPU instead: Set `DEVICE=cpu` in environment
- Close other GPU-intensive applications

**Port already in use:**
```bash
# Find process using port 8000
lsof -i :8000

# Kill process
kill -9 <PID>
```

### Frontend Issues

**Cannot connect to API:**
- Check backend is running: `curl http://localhost:8000/health`
- Verify proxy in package.json
- Check CORS settings in backend
- Ensure firewall allows port 8000

**Build fails:**
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
npm run build
```

## Security Considerations

1. **CORS**: Restrict to specific domains in production
2. **File Upload**: Implement file size limits and type validation
3. **Rate Limiting**: Add rate limiting to prevent abuse
4. **HTTPS**: Use SSL certificates in production
5. **Authentication**: Add authentication for sensitive deployments

## Testing

### Backend

```bash
cd api
python test_client.py
```

### Frontend

```bash
cd frontend
npm test
```

### Integration Test

```bash
# Start both servers
# Backend: http://localhost:8000
# Frontend: http://localhost:3000

# Upload test image through UI
# Verify prediction results
```

## Model Information

- **Architecture**: ResNet18 (Transfer Learning)
- **Dataset**: Food-101 (101 food categories)
- **Training Accuracy**: 76.32%
- **Input Size**: 224x224 RGB images
- **Preprocessing**: ImageNet normalization

## Support

For issues or questions:
- Check API docs: `http://localhost:8000/docs`
- Review logs for errors
- Verify model file exists and is valid
- Ensure all dependencies are installed

## Directory Structure

```
llm_prj/
├── notebooks/           # Training notebooks
│   └── food101_training_optimal.ipynb
├── api/                 # FastAPI backend
│   ├── main.py
│   ├── requirements.txt
│   ├── test_client.py
│   └── README.md
├── frontend/            # React frontend
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── README.md
├── outputs/
│   └── models/
│       └── best_model.pth
└── data/
    └── food-101/
```
