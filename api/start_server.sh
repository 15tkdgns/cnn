#!/bin/bash

echo "======================================"
echo "Food-101 API 서버 시작 중..."
echo "======================================"

# 현재 디렉토리 확인
cd "$(dirname "$0")"

# 필수 파일 확인
MODEL_PATH="../outputs/models/best_model.pth"

if [ ! -f "$MODEL_PATH" ]; then
    echo "⚠️  경고: 모델 파일을 찾을 수 없습니다: $MODEL_PATH"
    echo "   모델 없이 서버를 시작합니다 (정확도가 낮을 수 있습니다)"
    echo ""
fi

# 서버 시작
echo "🚀 서버 시작 중..."
echo "   URL: http://localhost:8000"
echo "   API 문서: http://localhost:8000/docs"
echo ""
echo "서버를 중지하려면 Ctrl+C를 누르세요"
echo "======================================"
echo ""

uvicorn main:app --host 0.0.0.0 --port 8000 --reload
