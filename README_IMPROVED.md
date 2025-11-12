# Food-101 Image Classification System

> ResNet18 기반 음식 이미지 분류 및 Grad-CAM 시각화 프로젝트

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![React](https://img.shields.io/badge/React-18.2-blue.svg)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 📋 목차

- [프로젝트 개요](#프로젝트-개요)
- [주요 기능](#주요-기능)
- [기술 스택](#기술-스택)
- [시스템 요구사항](#시스템-요구사항)
- [설치 및 설정](#설치-및-설정)
- [사용 방법](#사용-방법)
- [프로젝트 구조](#프로젝트-구조)
- [API 문서](#api-문서)
- [성능 벤치마크](#성능-벤치마크)
- [개발 가이드](#개발-가이드)
- [문제 해결](#문제-해결)
- [기여 방법](#기여-방법)
- [라이선스](#라이선스)

## 🎯 프로젝트 개요

Food-101 데이터셋을 사용한 음식 이미지 분류 시스템입니다. Transfer Learning 기반의 ResNet18 모델로 **76.32%의 테스트 정확도**를 달성했으며, Grad-CAM을 통해 AI의 판단 근거를 시각화할 수 있습니다.

### 핵심 특징

- ✅ **101가지 음식 분류** - apple_pie부터 waffles까지
- 🔍 **Grad-CAM 시각화** - AI가 어느 부분을 보고 판단했는지 확인
- 🎯 **YOLO 객체 탐지** - 80개 COCO 클래스 탐지 지원
- ⚡ **GPU 가속** - CUDA 최적화로 빠른 추론 속도 (~50ms)
- 🎨 **직관적인 UI** - 드래그 앤 드롭, 붙여넣기 지원
- 🔧 **프로덕션 레디** - 환경 변수, 로깅, 에러 처리 완비

## 🚀 주요 기능

### 1. 음식 이미지 분류

- ResNet18 기반 전이 학습
- Top-5 예측 결과 제공
- 신뢰도 점수 표시

### 2. Grad-CAM 히트맵

- AI 판단 근거 시각화
- 중요 영역을 색상으로 표시
- 빨간색: 중요도 높음, 파란색: 낮음

### 3. YOLO 객체 탐지

- 실시간 다중 객체 탐지
- 바운딩 박스 및 라벨 표시
- 신뢰도 임계값 조정 가능

## 🛠 기술 스택

### Backend
- **FastAPI** - 고성능 비동기 웹 프레임워크
- **PyTorch** - 딥러닝 모델 학습 및 추론
- **torchvision** - 사전학습 모델 및 이미지 변환
- **Ultralytics YOLO** - 객체 탐지
- **OpenCV** - 이미지 처리
- **Pydantic** - 데이터 검증

### Frontend
- **React 18** - UI 라이브러리
- **Axios** - HTTP 클라이언트
- **CSS3** - 스타일링

### Training
- **PyTorch** - 모델 훈련
- **scikit-learn** - 데이터 분할 및 평가
- **matplotlib** - 시각화

## 💻 시스템 요구사항

### 최소 요구사항
- Python 3.10+
- Node.js 18+
- 8GB RAM
- 10GB 저장 공간

### 권장 요구사항
- Python 3.10+
- Node.js 18+
- NVIDIA GPU (6GB+ VRAM)
- 16GB+ RAM
- 20GB+ 저장 공간

## 📦 설치 및 설정

### 1. 저장소 클론

```bash
git clone <repository-url>
cd llm_prj
```

### 2. 백엔드 설정

#### 2.1 Python 가상환경 생성

```bash
# Conda 사용
conda create -n food101 python=3.10
conda activate food101

# 또는 venv 사용
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows
```

#### 2.2 의존성 설치

```bash
cd api
pip install -r requirements.txt
```

#### 2.3 환경 변수 설정

```bash
# .env.example을 .env로 복사
cp .env.example .env

# .env 파일 편집 (필요 시)
nano .env
```

**주요 환경 변수:**

| 변수명 | 기본값 | 설명 |
|--------|--------|------|
| `HOST` | 0.0.0.0 | 서버 호스트 |
| `PORT` | 8000 | 서버 포트 |
| `MODEL_PATH` | ./outputs/models/best_model.pth | 모델 파일 경로 |
| `MAX_FILE_SIZE_MB` | 10 | 최대 파일 크기 (MB) |
| `ALLOWED_ORIGINS` | http://localhost:3000 | CORS 허용 origin |

### 3. 프론트엔드 설정

#### 3.1 의존성 설치

```bash
cd ../frontend
npm install
```

#### 3.2 환경 변수 설정

```bash
# .env.example을 .env로 복사
cp .env.example .env

# .env 파일 편집
nano .env
```

**주요 환경 변수:**

| 변수명 | 기본값 | 설명 |
|--------|--------|------|
| `REACT_APP_API_URL` | http://localhost:8000 | API 서버 URL |
| `REACT_APP_MAX_FILE_SIZE_MB` | 10 | 최대 파일 크기 |

### 4. 데이터셋 다운로드

```bash
cd ../scripts
python download_dataset.py
```

Food-101 데이터셋(~5GB)이 자동으로 다운로드됩니다.

### 5. 모델 훈련 (선택사항)

이미 훈련된 모델이 포함되어 있지만, 재훈련을 원할 경우:

```bash
cd ../notebooks
python food101_training.py
```

**훈련 시간:** GPU 기준 ~2.5시간 (10 에폭)

## 🎮 사용 방법

### 개발 모드 실행

#### Terminal 1: 백엔드 서버 시작

```bash
cd api
python -m uvicorn main_refactored:app --reload --host 0.0.0.0 --port 8000
```

또는 편의 스크립트 사용:

```bash
./start_backend.sh
```

**접속:** http://localhost:8000
**API 문서:** http://localhost:8000/docs

#### Terminal 2: 프론트엔드 서버 시작

```bash
cd frontend
npm start
```

또는:

```bash
./start_frontend.sh
```

**접속:** http://localhost:3000

### 프로덕션 배포

#### Docker 사용

```bash
# 백엔드
docker build -t food-classifier-api ./api
docker run -p 8000:8000 food-classifier-api

# 프론트엔드
docker build -t food-classifier-ui ./frontend
docker run -p 3000:80 food-classifier-ui
```

#### Nginx + Gunicorn

```bash
# Gunicorn으로 백엔드 실행
gunicorn api.main_refactored:app -w 4 -k uvicorn.workers.UvicornWorker

# 프론트엔드 빌드
cd frontend
npm run build

# Nginx로 서빙
nginx -c nginx.conf
```

## 📁 프로젝트 구조

```
llm_prj/
├── api/                        # 백엔드 (FastAPI)
│   ├── main.py                 # 기존 API 서버 (레거시)
│   ├── main_refactored.py      # 리팩토링된 API 서버 ⭐
│   ├── config.py               # 환경 설정 관리
│   ├── models.py               # Pydantic 모델
│   ├── utils.py                # 유틸리티 함수
│   ├── logger.py               # 로깅 설정
│   ├── gradcam.py              # Grad-CAM 구현
│   └── yolo_detector.py        # YOLO 래퍼
│
├── frontend/                   # 프론트엔드 (React)
│   ├── src/
│   │   ├── App.js              # 기존 메인 컴포넌트 (레거시)
│   │   ├── App_refactored.js   # 리팩토링된 메인 컴포넌트 ⭐
│   │   ├── components/         # 재사용 가능 컴포넌트
│   │   │   ├── UploadZone.js
│   │   │   ├── ModeSelector.js
│   │   │   ├── ClassificationResult.js
│   │   │   └── DetectionResult.js
│   │   ├── hooks/              # 커스텀 훅
│   │   │   ├── useImageUpload.js
│   │   │   └── usePrediction.js
│   │   └── services/           # API 서비스 레이어
│   │       └── api.js
│   └── package.json
│
├── notebooks/                  # 학습 스크립트
│   └── food101_training.py     # 모델 훈련 코드
│
├── outputs/                    # 모델 출력
│   ├── models/
│   │   └── best_model.pth      # 훈련된 모델 (76.32% 정확도)
│   └── images/                 # 시각화 결과
│
├── data/                       # 데이터셋
│   └── food-101/               # Food-101 데이터
│
├── scripts/                    # 유틸리티 스크립트
│   ├── download_dataset.py
│   ├── start_backend.sh
│   └── start_frontend.sh
│
├── .env.example                # 환경 변수 예제
└── README_IMPROVED.md          # 이 문서 ⭐
```

## 📚 API 문서

### 엔드포인트 목록

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/` | GET | 웹 인터페이스 또는 API 정보 |
| `/health` | GET | 헬스 체크 |
| `/classes` | GET | 101개 음식 클래스 목록 |
| `/predict` | POST | 음식 분류 (Top-5) |
| `/predict/gradcam` | POST | 음식 분류 + Grad-CAM |
| `/detect` | POST | YOLO 객체 탐지 |
| `/detect/classes` | GET | YOLO 클래스 목록 (80개) |
| `/docs` | GET | Swagger UI (대화형 API 문서) |

### 사용 예제

#### 1. 음식 분류 (cURL)

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@pizza.jpg"
```

**응답:**

```json
{
  "success": true,
  "prediction": {
    "class": "pizza",
    "class_id": 53,
    "confidence": 0.8523,
    "confidence_percent": "85.23%"
  },
  "top5": [
    {"rank": 1, "class": "pizza", "confidence": 0.8523, ...},
    {"rank": 2, "class": "lasagna", "confidence": 0.0823, ...},
    ...
  ]
}
```

#### 2. Grad-CAM 히트맵 (Python)

```python
import requests

with open('burger.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict/gradcam',
        files={'file': f}
    )

result = response.json()
heatmap_base64 = result['gradcam']['heatmap_image']
# <img src="data:image/png;base64,..." />
```

#### 3. YOLO 객체 탐지 (JavaScript)

```javascript
const formData = new FormData();
formData.append('file', fileObject);

const response = await fetch('http://localhost:8000/detect', {
  method: 'POST',
  body: formData
});

const result = await response.json();
console.log(`탐지된 객체: ${result.num_objects}개`);
```

자세한 API 문서는 http://localhost:8000/docs 참고

## 📊 성능 벤치마크

### 모델 정확도

| 데이터셋 | 정확도 | 손실 |
|---------|-------|------|
| 훈련 세트 | ~64% | 1.2 |
| 검증 세트 | 64.23% | 1.3 |
| 테스트 세트 | **76.32%** | - |

### 추론 속도 (RTX 3060 12GB)

| 작업 | 이미지 크기 | GPU | CPU |
|------|-----------|-----|-----|
| ResNet18 분류 | 224×224 | ~50ms | ~200ms |
| ResNet18 + Grad-CAM | 224×224 | ~80ms | ~350ms |
| YOLO11n 탐지 | 640×640 | ~30ms | ~150ms |

### 메모리 사용량

| 컴포넌트 | GPU 메모리 | CPU 메모리 |
|---------|-----------|-----------|
| ResNet18 (추론) | ~60MB | ~300MB |
| YOLO11n (추론) | ~80MB | ~150MB |
| FastAPI 서버 | - | ~200MB |

## 👨‍💻 개발 가이드

### 코드 스타일

- **Python:** PEP 8 준수, Black 포매터 사용
- **JavaScript:** ES6+, Prettier 사용

### 테스트 실행

```bash
# 백엔드 테스트
cd api
pytest tests/

# 프론트엔드 테스트
cd frontend
npm test
```

### 새로운 기능 추가

1. Feature 브랜치 생성: `git checkout -b feature/your-feature`
2. 코드 작성 및 테스트
3. 커밋: `git commit -m "feat: add your feature"`
4. Pull Request 생성

## 🐛 문제 해결

### 문제: "모델 파일을 찾을 수 없습니다"

**해결:**

```bash
# 모델 파일 경로 확인
ls outputs/models/best_model.pth

# 없으면 재훈련
python notebooks/food101_training.py
```

### 문제: "CUDA out of memory"

**해결:**

```python
# config.py 또는 .env에서 배치 크기 감소
BATCH_SIZE = 64  # 기본값: 128
```

### 문제: "CORS 에러"

**해결:**

```bash
# .env 파일에서 ALLOWED_ORIGINS 설정
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:3001
```

### 문제: 프론트엔드가 API에 연결 안됨

**해결:**

```bash
# 1. 백엔드가 실행 중인지 확인
curl http://localhost:8000/health

# 2. 프론트엔드 .env 파일 확인
cat frontend/.env
# REACT_APP_API_URL=http://localhost:8000

# 3. 브라우저 콘솔에서 에러 확인
```

## 🤝 기여 방법

1. 이 저장소를 Fork
2. Feature 브랜치 생성 (`git checkout -b feature/AmazingFeature`)
3. 변경사항 커밋 (`git commit -m 'feat: Add AmazingFeature'`)
4. 브랜치에 Push (`git push origin feature/AmazingFeature`)
5. Pull Request 생성

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 📧 연락처

- 이슈 리포트: [GitHub Issues](https://github.com/yourusername/food-classifier/issues)
- 이메일: your.email@example.com

## 🙏 감사의 말

- [Food-101 Dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
- [PyTorch](https://pytorch.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)

---

**Made with ❤️ by Your Team**
