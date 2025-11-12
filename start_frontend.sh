#!/bin/bash
# Start Food-101 Classification Frontend

echo "🎨 Starting Food-101 React Frontend..."
echo "======================================"
echo ""

# Navigate to frontend directory
cd "$(dirname "$0")/frontend"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
    echo ""
fi

# Start the development server
echo "✅ Starting React development server on http://localhost:3000"
echo "   The app will open in your browser automatically"
echo ""
echo "⚠️  Make sure the backend API is running on http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop the server"
echo "======================================"
echo ""

npm start
