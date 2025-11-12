#!/bin/bash
# Start Food-101 Classification API Backend

echo "🚀 Starting Food-101 API Backend..."
echo "=================================="
echo ""

# Navigate to API directory
cd "$(dirname "$0")/api"

# Check if model exists
if [ ! -f "../outputs/models/best_model.pth" ]; then
    echo "⚠️  Warning: Model file not found at ../outputs/models/best_model.pth"
    echo "    The API will start but predictions may not work correctly."
    echo "    Please train the model first using the Jupyter notebook."
    echo ""
fi

# Check if dependencies are installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
    echo ""
fi

# Start the server
echo "✅ Starting FastAPI server on http://localhost:8000"
echo "   API Documentation: http://localhost:8000/docs"
echo "   Health Check: http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=================================="
echo ""

python main.py
