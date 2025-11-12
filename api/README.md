# Food-101 Image Classification API

ResNet18 기반 음식 이미지 분류 FastAPI 서버

## 📋 목차
- [기능](#기능)
- [설치](#설치)
- [사용법](#사용법)
- [API 엔드포인트](#api-엔드포인트)
- [예제](#예제)

## ✨ 기능

- **이미지 분류**: 음식 이미지를 업로드하여 101개 클래스 중 하나로 분류
- **Top-5 예측**: 가장 확률이 높은 5개 클래스 반환
- **배치 처리**: 여러 이미지를 한 번에 처리
- **CORS 지원**: 프론트엔드에서 직접 호출 가능
- **헬스 체크**: 서버 상태 모니터링

## 🚀 설치

### 1. 의존성 설치

```bash
cd api
pip install -r requirements.txt
```

### 2. 필수 파일 확인

다음 파일들이 필요합니다:
- `../outputs/models/best_model.pth` - 훈련된 모델 가중치
- `../data/food-101/food-101/meta/classes.txt` - 클래스 목록

## 💻 사용법

### 서버 시작

#### 방법 1: Python으로 직접 실행
```bash
cd api
python main.py
```

#### 방법 2: Uvicorn 사용 (개발 모드)
```bash
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### 방법 3: 프로덕션 모드
```bash
cd api
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

서버가 시작되면 다음 URL에서 접근 가능합니다:
- API 서버: http://localhost:8000
- API 문서 (Swagger UI): http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 📡 API 엔드포인트

### 1. 루트
```
GET /
```
API 정보 및 사용 가능한 엔드포인트 목록 반환

**응답 예시:**
```json
{
  "message": "Food-101 Image Classification API",
  "version": "1.0.0",
  "endpoints": {
    "predict": "POST /predict - 이미지 업로드 및 분류",
    "health": "GET /health - 서버 상태 확인",
    "classes": "GET /classes - 분류 가능한 음식 목록"
  }
}
```

### 2. 헬스 체크
```
GET /health
```
서버 및 모델 상태 확인

**응답 예시:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "num_classes": 101
}
```

### 3. 클래스 목록
```
GET /classes
```
분류 가능한 음식 클래스 목록 반환

**응답 예시:**
```json
{
  "total": 101,
  "classes": [
    "apple_pie",
    "baby_back_ribs",
    "baklava",
    ...
  ]
}
```

### 4. 이미지 예측
```
POST /predict
```
이미지를 업로드하여 음식 종류 예측

**요청:**
- Content-Type: `multipart/form-data`
- Body: `file` (이미지 파일)

**응답 예시:**
```json
{
  "success": true,
  "prediction": {
    "class": "apple_pie",
    "class_id": 0,
    "confidence": 0.9234,
    "confidence_percent": "92.34%"
  },
  "top5": [
    {
      "rank": 1,
      "class": "apple_pie",
      "class_id": 0,
      "confidence": 0.9234,
      "confidence_percent": "92.34%"
    },
    {
      "rank": 2,
      "class": "baklava",
      "class_id": 2,
      "confidence": 0.0543,
      "confidence_percent": "5.43%"
    },
    ...
  ]
}
```

### 5. 배치 예측
```
POST /predict/batch
```
여러 이미지를 한 번에 예측 (최대 10개)

**요청:**
- Content-Type: `multipart/form-data`
- Body: `files` (이미지 파일 리스트)

**응답 예시:**
```json
{
  "success": true,
  "total": 3,
  "results": [
    {
      "filename": "image1.jpg",
      "prediction": {
        "class": "apple_pie",
        "class_id": 0,
        "confidence": 0.9234,
        "confidence_percent": "92.34%"
      }
    },
    ...
  ]
}
```

## 📝 예제

### Python (requests)

```python
import requests

# 단일 이미지 예측
with open("food_image.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post("http://localhost:8000/predict", files=files)
    result = response.json()

print(f"예측 결과: {result['prediction']['class']}")
print(f"확신도: {result['prediction']['confidence_percent']}")
```

### cURL

```bash
# 헬스 체크
curl http://localhost:8000/health

# 클래스 목록
curl http://localhost:8000/classes

# 이미지 예측
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@food_image.jpg"
```

### JavaScript (Fetch API)

```javascript
// 이미지 파일 업로드
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8000/predict', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => {
  console.log('예측 결과:', data.prediction.class);
  console.log('확신도:', data.prediction.confidence_percent);
});
```

### 테스트 클라이언트 사용

```bash
cd api
python test_client.py
```

## 🔧 고급 설정

### 환경 변수

`.env` 파일을 생성하여 설정을 커스터마이즈할 수 있습니다:

```env
# 서버 설정
HOST=0.0.0.0
PORT=8000
WORKERS=4

# 모델 경로
MODEL_PATH=../outputs/models/best_model.pth
CLASSES_PATH=../data/food-101/food-101/meta/classes.txt

# 디바이스
DEVICE=cuda  # 또는 cpu
```

### Docker 실행 (선택사항)

```dockerfile
# Dockerfile 예시
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Docker 이미지 빌드 및 실행
docker build -t food101-api .
docker run -p 8000:8000 food101-api
```

## 📊 성능

- **모델**: ResNet18 (전이학습)
- **테스트 정확도**: 76.32%
- **클래스 수**: 101개
- **추론 속도**: ~50ms/image (GPU), ~200ms/image (CPU)

## ❓ 문제 해결

### 모델을 찾을 수 없음
```
⚠️  모델 파일을 찾을 수 없습니다
```
**해결방법**: 모델이 `../outputs/models/best_model.pth`에 있는지 확인

### CUDA out of memory
```
RuntimeError: CUDA out of memory
```
**해결방법**:
1. 배치 크기 줄이기
2. CPU 모드로 전환: `DEVICE=cpu python main.py`

### 포트가 이미 사용 중
```
ERROR: [Errno 98] Address already in use
```
**해결방법**: 다른 포트 사용
```bash
uvicorn main:app --port 8001
```

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🤝 기여

버그 리포트나 기능 제안은 이슈로 등록해주세요.
