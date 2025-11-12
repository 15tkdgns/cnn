# Food-101 Image Classification Project

딥러닝을 활용한 101가지 음식 이미지 분류 시스템 (Full-Stack Application)

## 프로젝트 개요

- **목표**: ResNet18 기반 전이학습을 통한 음식 이미지 자동 분류
- **데이터셋**: Food-101 (101개 클래스, 101,000장 이미지)
- **프레임워크**: PyTorch (Model), FastAPI (Backend), React (Frontend)
- **주요 기술**: Transfer Learning, Mixed Precision Training, GPU 최적화, REST API, ChatGPT-style UI
- **달성 정확도**: 76.32% (Test Set)

## 시스템 구성

이 프로젝트는 3개의 주요 컴포넌트로 구성됩니다:

1. **모델 훈련** (`notebooks/`) - ResNet18 전이학습 (76.32% 정확도)
2. **백엔드 API** (`api/`) - FastAPI 기반 REST API 서버
3. **프론트엔드** (`frontend/`) - ChatGPT 스타일의 React 웹 인터페이스

## 📊 데이터 흐름 및 통신 구조

이 프로젝트의 각 컴포넌트가 어떻게 데이터를 주고받는지 이해하려면 **[DATA_FLOW.md](DATA_FLOW.md)** 문서를 참조하세요.

### 간단 요약

```
사용자 이미지 업로드
    ↓
React Frontend (localhost:3000)
    ↓ HTTP POST (multipart/form-data)
FastAPI Backend (localhost:8000)
    ↓ PyTorch 추론
ResNet18 / YOLO 모델 (GPU)
    ↓ JSON 응답
사용자에게 결과 표시
```

**주요 통신 방식**:
- **학습 → 모델**: PyTorch checkpoint (best_model.pth)
- **클라이언트 → 서버**: HTTP/JSON REST API
- **서버 → 모델**: Python 함수 호출 (GPU 메모리)

자세한 내용은 [DATA_FLOW.md](DATA_FLOW.md)를 확인하세요.

## 프로젝트 구조

```
llm_prj/
├── README.md                       # 프로젝트 설명 (본 파일)
├── DATA_FLOW.md                    # 데이터 흐름 및 통신 구조 (상세)
├── DEPLOYMENT.md                   # 배포 가이드
├── start_backend.sh                # 백엔드 시작 스크립트
├── start_frontend.sh               # 프론트엔드 시작 스크립트
├── requirements.txt                # Python 패키지 의존성
│
├── notebooks/                      # Jupyter Notebooks
│   ├── food101_training.py         # 훈련 스크립트
│   └── food101_training_optimal.ipynb  # 메인 훈련 노트북
│
├── api/                            # FastAPI 백엔드
│   ├── main.py                     # FastAPI 애플리케이션
│   ├── requirements.txt            # API 의존성
│   ├── test_client.py              # API 테스트 스크립트
│   ├── start_server.sh             # 서버 시작 스크립트
│   ├── static/                     # 정적 파일
│   │   └── index.html              # 기본 웹 인터페이스
│   └── README.md                   # API 문서
│
├── frontend/                       # React 프론트엔드
│   ├── public/
│   │   └── index.html              # HTML 쉘
│   ├── src/
│   │   ├── index.js                # React 엔트리포인트
│   │   ├── index.css               # 글로벌 스타일
│   │   ├── App.js                  # 메인 컴포넌트
│   │   └── App.css                 # ChatGPT 스타일 CSS
│   ├── package.json                # NPM 의존성
│   └── README.md                   # 프론트엔드 문서
│
├── data/                           # 데이터셋 관련
│   ├── dataset_path.txt            # 데이터셋 경로
│   └── food-101/                   # Food-101 데이터셋
│
├── outputs/                        # 훈련 결과물
│   ├── models/
│   │   └── best_model.pth          # 학습된 모델 (76.32%)
│   └── images/                     # 결과 시각화 이미지
│
├── scripts/                        # 유틸리티 스크립트
│   ├── download_dataset.py         # 데이터셋 다운로드
│   └── explore_dataset.py          # 데이터 탐색
│
└── docs/                           # 문서 및 발표 자료
    ├── presentation_script.txt     # 발표 스크립트
    └── qna_material.txt            # Q&A 자료
```

## 빠른 시작

### 사전 요구사항

- Python 3.8+
- Node.js 14+
- NVIDIA GPU (권장, CPU도 가능)
- 훈련된 모델 파일: `outputs/models/best_model.pth`

### 1. 백엔드 API 실행

```bash
# 방법 1: 편리한 스크립트 사용
chmod +x start_backend.sh
./start_backend.sh

# 방법 2: 수동 실행
cd api
pip install -r requirements.txt
python main.py
```

백엔드는 **http://localhost:8000** 에서 실행됩니다.

### 2. 프론트엔드 실행

새 터미널을 열고:

```bash
# 방법 1: 편리한 스크립트 사용
chmod +x start_frontend.sh
./start_frontend.sh

# 방법 2: 수동 실행
cd frontend
npm install
npm start
```

프론트엔드는 **http://localhost:3000** 에서 자동으로 열립니다.

### 3. 애플리케이션 사용

1. 브라우저에서 `http://localhost:3000` 열기
2. 음식 이미지 업로드 (드래그 앤 드롭, 클릭, 또는 Ctrl+V)
3. "분석하기" 클릭하여 분류 수행
4. 예측된 음식 종류와 Top-5 결과 확인

### 4. 모델 훈련 (선택사항)

모델 파일이 없는 경우:

```bash
cd notebooks
jupyter notebook food101_training_optimal.ipynb
# 모든 셀을 실행하여 모델 훈련
```

## 주요 기능

### 모델 훈련
- **모델 아키텍처**: ResNet18 (ImageNet 사전학습)
- **데이터셋**: Food-101 (101,000 이미지, 101 클래스)
- **훈련 정확도**: 76.32% (Test Set)
- **최적 하이퍼파라미터**: 10 epochs, learning rate 5e-4
- **GPU 최적화**: Mixed Precision, DataLoader 최적화
- **훈련 시간**: GPU에서 약 2-3시간

### 백엔드 API (FastAPI)
- **프레임워크**: FastAPI with automatic OpenAPI docs
- **엔드포인트**:
  - `POST /predict` - 단일 이미지 분류
  - `POST /predict/batch` - 배치 분류 (최대 10개)
  - `GET /health` - 헬스 체크
  - `GET /classes` - 101개 음식 클래스 목록
  - `GET /docs` - Swagger UI 문서
- **기능**: CORS 지원, Top-5 예측, GPU/CPU 지원

### 프론트엔드 (React)
- **디자인**: ChatGPT 스타일 미니멀 UI
- **기능**:
  - 드래그 앤 드롭 이미지 업로드
  - 클릭하여 업로드
  - 클립보드 붙여넣기 (Ctrl+V)
  - 실시간 이미지 미리보기
  - Top-5 예측 결과 (신뢰도 바)
  - 반응형 모바일 디자인
- **기술**: React 18, Axios, CSS3

## 성능

- **훈련 속도**: 에폭당 약 1-1.5분 (GPU 최적화 후)
- **GPU 사용률**: 85-95% (최적화 전 20-40% 대비 크게 향상)
- **예상 정확도**: 70-85% (테스트 세트 기준)

## 시스템 요구사항

### 최소 사양
- **GPU**: NVIDIA GPU 6GB+ (GTX 1060 6GB 이상)
- **RAM**: 16GB
- **저장공간**: 5GB (데이터셋 포함)
- **CUDA**: 11.0 이상

### 권장 사양
- **GPU**: NVIDIA RTX 3060 12GB 이상
- **RAM**: 32GB
- **저장공간**: 10GB

## 주요 파일 설명

### notebooks/food101_training.ipynb
메인 훈련 노트북으로 다음 기능을 포함합니다:
1. 환경 설정 및 라이브러리 임포트
2. 데이터 로딩 및 전처리
3. 모델 구성 (ResNet18)
4. 하이퍼파라미터 탐색 실험
5. 결과 시각화 및 평가

### 데이터 증강 제거
이 버전에서는 하이퍼파라미터 최적화에 집중하기 위해 데이터 증강을 사용하지 않습니다.
기본 전처리(Resize, CenterCrop, Normalize)만 적용합니다.

## 훈련 모니터링

GPU 사용률 실시간 확인:
```bash
# 간단한 모니터링
watch -n 1 nvidia-smi

# 상세 모니터링
nvidia-smi dmon -s pucvmet -d 1
```

## 문제 해결

### GPU 메모리 부족
배치 크기를 줄입니다:
```python
BATCH_SIZE = 128  # 또는 64
```

### 데이터 로딩 느림
워커 수를 조정합니다:
```python
NUM_WORKERS = 4  # 또는 2
```

### 시스템 과부하
Prefetch factor를 줄입니다:
```python
prefetch_factor = 2
```

## 참고 자료

### 문서
- [발표 스크립트](docs/presentation_script.txt)
- [Q&A 자료](docs/qna_material.txt)

### 논문 및 리소스
- [Food-101 Dataset Paper](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [PyTorch 공식 문서](https://pytorch.org/docs/)

## 라이선스

이 프로젝트는 교육 및 연구 목적으로 사용됩니다.
Food-101 데이터셋은 ETH Zurich의 라이선스를 따릅니다.

## 기여

버그 리포트나 개선 제안은 이슈로 등록해주세요.

## API 사용 예제

### Python
```python
import requests

with open('pizza.jpg', 'rb') as f:
    response = requests.post('http://localhost:8000/predict', files={'file': f})
    result = response.json()
    print(f"예측: {result['prediction']['class']}")
    print(f"신뢰도: {result['prediction']['confidence_percent']}")
```

### cURL
```bash
curl -X POST "http://localhost:8000/predict" -F "file=@pizza.jpg"
```

### JavaScript
```javascript
const formData = new FormData();
formData.append('file', imageFile);
const response = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  body: formData
});
const result = await response.json();
```

## 배포

자세한 배포 가이드는 [DEPLOYMENT.md](DEPLOYMENT.md)를 참조하세요.

- Docker 배포
- Nginx 설정
- Systemd 서비스
- 환경 변수 설정
- 프로덕션 최적화

## 변경 이력

### v3.0 (2025-11-11)
- **Full-Stack Application 완성**
- FastAPI 백엔드 추가 (REST API)
- React 프론트엔드 추가 (ChatGPT 스타일)
- 배포 가이드 및 문서화
- 편리한 시작 스크립트 제공

### v2.0 (2024-11-10)
- GPU 최적화 (배치 크기 증가, DataLoader 최적화)
- 하이퍼파라미터 자동 탐색 기능 추가
- 프로젝트 구조 재정리
- 76.32% 정확도 달성

### v1.0 (2024-11-09)
- 초기 버전
- ResNet18 기반 전이학습 구현
- 기본 훈련 파이프라인

## 연락처

프로젝트 관련 문의사항이 있으시면 이슈를 생성해주세요.
