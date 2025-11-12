# Food-101 Classifier - React Frontend

ChatGPT-inspired minimal design for Food-101 image classification.

## Features

- **Drag & Drop**: Drag images directly onto the upload zone
- **Click Upload**: Click the upload zone to browse files
- **Paste Support**: Press Ctrl+V to paste images from clipboard
- **Real-time Preview**: See your image before analyzing
- **Top-5 Predictions**: View the top 5 most likely food classifications
- **Progress Bars**: Visual confidence indicators
- **Responsive Design**: Works on desktop and mobile devices

## Installation

```bash
npm install
```

## Development

Start the development server:

```bash
npm start
```

The app will open at `http://localhost:3000`

## Backend Connection

The app connects to the FastAPI backend at `http://localhost:8000` (configured in package.json proxy).

Make sure the backend is running before using the frontend:

```bash
cd ../api
pip install -r requirements.txt
python main.py
```

Or use uvicorn:

```bash
cd ../api
uvicorn main:app --reload
```

## Build

Create production build:

```bash
npm run build
```

The build files will be in the `build/` directory.

## Design

Inspired by ChatGPT's clean and minimal interface:
- White backgrounds with subtle borders
- ChatGPT green (#10a37f) for primary actions
- Smooth transitions and hover effects
- Centered content layout
- Sticky header and footer

## API Integration

Uses Axios to communicate with the FastAPI backend:
- `POST /predict` - Single image classification
- Returns prediction with confidence and Top-5 results

## File Structure

```
frontend/
├── package.json          # Dependencies and scripts
├── public/
│   └── index.html       # HTML shell
└── src/
    ├── index.js         # React entry point
    ├── index.css        # Global styles
    ├── App.js           # Main component
    └── App.css          # ChatGPT-style CSS
```

## Environment Variables

Create `.env` file to customize API URL:

```
REACT_APP_API_URL=http://localhost:8000
```

Default is `http://localhost:8000` if not specified.
