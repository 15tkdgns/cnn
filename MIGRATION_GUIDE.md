# 마이그레이션 가이드

> 기존 코드에서 리팩토링된 코드로 전환하는 방법

## 📋 목차

- [개요](#개요)
- [주요 변경사항](#주요-변경사항)
- [마이그레이션 단계](#마이그레이션-단계)
- [설정 파일 변환](#설정-파일-변환)
- [코드 변경 사항](#코드-변경-사항)
- [테스트 방법](#테스트-방법)
- [롤백 방법](#롤백-방법)

## 🎯 개요

이 가이드는 기존의 모놀리식 코드를 **모듈화되고 유지보수 가능한 구조**로 전환하는 방법을 안내합니다.

### 마이그레이션 목적

- ✅ **가독성 향상** - 코드 구조 명확화
- ✅ **유지보수성 증가** - 모듈화된 컴포넌트
- ✅ **보안 강화** - 파일 검증, CORS 제한
- ✅ **에러 처리 개선** - 구조화된 로깅
- ✅ **설정 중앙화** - 환경 변수 관리

## 🔄 주요 변경사항

### 백엔드 (FastAPI)

#### Before (기존)

```
api/
├── main.py              # 모든 로직이 하나의 파일 (619줄)
├── gradcam.py
└── yolo_detector.py
```

#### After (리팩토링)

```
api/
├── main_refactored.py   # 메인 서버 (모듈화)
├── config.py            # 설정 관리 ⭐ NEW
├── models.py            # Pydantic 모델 ⭐ NEW
├── utils.py             # 유틸리티 함수 ⭐ NEW
├── logger.py            # 로깅 설정 ⭐ NEW
├── gradcam.py
└── yolo_detector.py
```

#### 주요 개선사항

| 항목 | 기존 | 리팩토링 |
|------|------|----------|
| 설정 관리 | 하드코딩 | 환경 변수 + config.py |
| 타입 안전성 | 없음 | Pydantic 모델 |
| 파일 검증 | 없음 | 크기/타입 검증 |
| 로깅 | 기본 logging | 구조화된 로깅 + 회전 |
| CORS | `allow_origins=["*"]` | 설정 파일에서 관리 |

### 프론트엔드 (React)

#### Before (기존)

```
frontend/src/
└── App.js               # 모든 로직이 하나의 컴포넌트 (453줄)
```

#### After (리팩토링)

```
frontend/src/
├── App_refactored.js    # 메인 컴포넌트 (모듈화)
├── components/          # 재사용 가능 컴포넌트 ⭐ NEW
│   ├── UploadZone.js
│   ├── ModeSelector.js
│   ├── ClassificationResult.js
│   └── DetectionResult.js
├── hooks/               # 커스텀 훅 ⭐ NEW
│   ├── useImageUpload.js
│   └── usePrediction.js
└── services/            # API 서비스 레이어 ⭐ NEW
    └── api.js
```

#### 주요 개선사항

| 항목 | 기존 | 리팩토링 |
|------|------|----------|
| 컴포넌트 | 1개 거대 컴포넌트 | 5개 작은 컴포넌트 |
| 상태 관리 | useState 8개 | 커스텀 훅 2개 |
| API 호출 | 인라인 axios | 서비스 레이어 |
| 에러 처리 | 기본 try-catch | 인터셉터 + 사용자 메시지 |

## 🚀 마이그레이션 단계

### Step 1: 백업 생성

```bash
# 전체 프로젝트 백업
cd /root
tar -czf llm_prj_backup_$(date +%Y%m%d).tar.gz llm_prj/

# 또는 Git 사용 시
cd llm_prj
git add .
git commit -m "backup: 마이그레이션 전 백업"
git tag -a v1.0.0 -m "마이그레이션 전 버전"
```

### Step 2: 환경 변수 설정

#### 2.1 백엔드 환경 변수

```bash
cd /root/llm_prj

# .env.example을 .env로 복사
cp .env.example .env

# 기존 설정을 .env로 이전
nano .env
```

**변환 예시:**

| 기존 코드 (main.py) | 새 환경 변수 (.env) |
|-------------------|---------------------|
| `app.add_middleware(..., allow_origins=["*"])` | `ALLOWED_ORIGINS=http://localhost:3000` |
| `model_path = Path("...") / "best_model.pth"` | `MODEL_PATH=./outputs/models/best_model.pth` |
| (하드코딩) 10MB 제한 | `MAX_FILE_SIZE_MB=10` |

#### 2.2 프론트엔드 환경 변수

```bash
cd frontend

# .env.example을 .env로 복사
cp .env.example .env

# API URL 설정
echo "REACT_APP_API_URL=http://localhost:8000" >> .env
```

### Step 3: 의존성 설치 (새 패키지)

#### 백엔드

```bash
cd api
pip install pydantic-settings  # config.py에 필요
```

#### 프론트엔드

```bash
cd frontend
# 추가 패키지 없음 (기존 axios 사용)
```

### Step 4: 백엔드 전환

#### 옵션 A: 점진적 마이그레이션 (권장)

두 버전을 동시에 실행하면서 테스트:

```bash
# Terminal 1: 기존 서버 (포트 8000)
python -m uvicorn api.main:app --reload --port 8000

# Terminal 2: 새 서버 (포트 8001)
python -m uvicorn api.main_refactored:app --reload --port 8001

# 프론트엔드에서 .env 수정하여 테스트
# REACT_APP_API_URL=http://localhost:8001
```

테스트 완료 후 main_refactored.py를 main.py로 대체:

```bash
cd api

# 백업
mv main.py main_legacy.py

# 새 버전을 메인으로 설정
mv main_refactored.py main.py

# 서버 재시작
python -m uvicorn api.main:app --reload
```

#### 옵션 B: 직접 교체

```bash
cd api

# 기존 파일 백업
mv main.py main_legacy.py

# 새 파일을 메인으로
cp main_refactored.py main.py

# Import 경로 수정
sed -i 's/from \.config/from config/g' main.py
sed -i 's/from \.models/from models/g' main.py
```

### Step 5: 프론트엔드 전환

#### 옵션 A: 점진적 마이그레이션

```bash
cd frontend/src

# 기존 파일 백업
mv App.js App_legacy.js

# 새 버전을 메인으로
cp App_refactored.js App.js

# 서버 재시작
npm start
```

#### 옵션 B: 라우팅으로 병행 운영

```javascript
// src/index.js
import AppLegacy from './App_legacy';
import AppRefactored from './App_refactored';

const isRefactored = process.env.REACT_APP_USE_REFACTORED === 'true';
const App = isRefactored ? AppRefactored : AppLegacy;

root.render(<App />);
```

### Step 6: Import 경로 업데이트

#### 백엔드

```python
# main.py에서
from config import settings           # ⭐ NEW
from models import ClassificationResponse  # ⭐ NEW
from utils import load_classes        # ⭐ NEW
from logger import api_logger         # ⭐ NEW
```

#### 프론트엔드

```javascript
// App.js에서
import api from './services/api';                       // ⭐ NEW
import useImageUpload from './hooks/useImageUpload';   // ⭐ NEW
import usePrediction from './hooks/usePrediction';     // ⭐ NEW
```

## 🔧 설정 파일 변환

### 백엔드: 하드코딩 → 환경 변수

| 항목 | 기존 (main.py) | 새 (.env + config.py) |
|------|--------------|---------------------|
| 모델 경로 | `Path(...) / "best_model.pth"` | `MODEL_PATH=./outputs/models/best_model.pth` |
| CORS | `allow_origins=["*"]` | `ALLOWED_ORIGINS=http://localhost:3000` |
| 파일 크기 | (없음) | `MAX_FILE_SIZE_MB=10` |
| 로그 레벨 | `logging.INFO` | `LOG_LEVEL=INFO` |

**변환 스크립트:**

```bash
# 기존 main.py에서 설정 추출
grep -E "(allow_origins|model_path|Path)" api/main.py > config_review.txt

# .env 파일 생성
cat > api/.env << EOF
HOST=0.0.0.0
PORT=8000
MODEL_PATH=./outputs/models/best_model.pth
ALLOWED_ORIGINS=http://localhost:3000
MAX_FILE_SIZE_MB=10
EOF
```

### 프론트엔드: 하드코딩 → 환경 변수

| 항목 | 기존 (App.js) | 새 (.env) |
|------|------------|-----------|
| API URL | `const API_URL = 'http://localhost:8000'` | `REACT_APP_API_URL=http://localhost:8000` |
| 앱 이름 | `<h1>Food Classifier</h1>` | `REACT_APP_NAME=Food Classifier` |

## 📝 코드 변경 사항

### 백엔드: API 엔드포인트

#### Before

```python
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    # ...
```

#### After

```python
from utils import validate_and_load_image

@app.post("/predict", response_model=ClassificationResponse)
async def predict(file: UploadFile = File(...)):
    image = await validate_and_load_image(
        file,
        settings.max_file_size_bytes,
        settings.allowed_file_types_list
    )
    # ...
```

**주요 변경점:**

1. `validate_and_load_image()` 유틸리티 함수 사용
2. `response_model` 추가 (타입 안전성)
3. 설정 값은 `settings` 객체에서 가져옴

### 프론트엔드: API 호출

#### Before

```javascript
const response = await axios.post(`${API_URL}/predict`, formData, {
  headers: { 'Content-Type': 'multipart/form-data' }
});
```

#### After

```javascript
import api from './services/api';

const response = await api.predictFood(image);
```

**주요 변경점:**

1. API 호출이 서비스 레이어로 추상화
2. 에러 처리가 인터셉터에서 자동 처리
3. 타임아웃, 재시도 로직 포함

## 🧪 테스트 방법

### 1. 기능 테스트 체크리스트

#### 백엔드

- [ ] 서버가 정상적으로 시작되는가?
  ```bash
  curl http://localhost:8000/health
  ```
- [ ] 환경 변수가 올바르게 로드되는가?
  ```bash
  python -c "from api.config import settings; print(settings.model_path)"
  ```
- [ ] 파일 크기 제한이 작동하는가?
  ```bash
  # 11MB 파일 생성
  dd if=/dev/zero of=large.jpg bs=1M count=11
  curl -X POST http://localhost:8000/predict -F "file=@large.jpg"
  # 예상: 413 Payload Too Large
  ```
- [ ] CORS가 올바르게 설정되었는가?
  ```bash
  curl -H "Origin: http://localhost:3000" -v http://localhost:8000/health
  # 헤더에 Access-Control-Allow-Origin 확인
  ```

#### 프론트엔드

- [ ] 이미지 업로드가 작동하는가?
- [ ] 드래그 앤 드롭이 작동하는가?
- [ ] 붙여넣기가 작동하는가?
- [ ] 음식 분류가 정상 작동하는가?
- [ ] Grad-CAM 히트맵이 표시되는가?
- [ ] YOLO 객체 탐지가 작동하는가?
- [ ] 에러 메시지가 올바르게 표시되는가?

### 2. 통합 테스트

```bash
# 백엔드 + 프론트엔드 동시 실행
# Terminal 1
cd api
python -m uvicorn main:app --reload

# Terminal 2
cd frontend
npm start

# 브라우저에서 http://localhost:3000 접속
# 모든 기능 테스트
```

### 3. 성능 테스트

```bash
# Apache Bench로 부하 테스트
ab -n 100 -c 10 http://localhost:8000/health

# 예상 결과:
# - Requests per second: > 100 req/sec
# - 실패율: 0%
```

## 🔄 롤백 방법

문제가 발생할 경우 빠르게 이전 버전으로 롤백:

### 옵션 1: 파일 백업으로 복원

```bash
cd /root/llm_prj

# 백엔드 롤백
cd api
mv main.py main_failed.py
mv main_legacy.py main.py

# 프론트엔드 롤백
cd ../frontend/src
mv App.js App_failed.js
mv App_legacy.js App.js

# 서버 재시작
```

### 옵션 2: Git 태그로 롤백

```bash
cd /root/llm_prj

# 마이그레이션 전 버전으로 복원
git checkout v1.0.0

# 또는 특정 커밋으로
git reset --hard <commit-hash>
```

### 옵션 3: 전체 백업 복원

```bash
cd /root

# 백업 압축 해제
tar -xzf llm_prj_backup_20241112.tar.gz

# 기존 폴더 교체
rm -rf llm_prj
mv llm_prj_backup llm_prj
```

## ⚠️ 주의사항

### 1. 데이터베이스 마이그레이션

현재 프로젝트는 데이터베이스를 사용하지 않지만, 향후 추가 시:

```python
# 마이그레이션 전
alembic revision --autogenerate -m "migration_v2"
alembic upgrade head
```

### 2. 프로덕션 배포 시

```bash
# 환경 변수 확인
printenv | grep REACT_APP
printenv | grep ALLOWED_ORIGINS

# 프로덕션 빌드
npm run build

# Gunicorn 워커 수 조정
gunicorn -w 4 -k uvicorn.workers.UvicornWorker api.main:app
```

### 3. 로그 파일 관리

```bash
# 로그 디렉토리 생성
mkdir -p logs

# 로그 회전 확인
ls -lh logs/

# 오래된 로그 삭제 (7일 이상)
find logs/ -name "*.log*" -mtime +7 -delete
```

## 📚 추가 리소스

- [FastAPI 공식 문서](https://fastapi.tiangolo.com/)
- [Pydantic 설정 관리](https://docs.pydantic.dev/latest/usage/settings/)
- [React 커스텀 훅](https://react.dev/learn/reusing-logic-with-custom-hooks)
- [Axios 인터셉터](https://axios-http.com/docs/interceptors)

## 🆘 문제 해결

### Q: "ModuleNotFoundError: No module named 'pydantic_settings'"

**A:**

```bash
pip install pydantic-settings
```

### Q: "ImportError: cannot import name 'settings' from 'config'"

**A:**

```bash
# config.py 파일이 api/ 디렉토리에 있는지 확인
ls api/config.py

# Python 경로 확인
python -c "import sys; print(sys.path)"
```

### Q: 프론트엔드에서 "Cannot find module './services/api'"

**A:**

```bash
# 파일 존재 확인
ls frontend/src/services/api.js

# 없으면 생성
mkdir -p frontend/src/services
cp api.js frontend/src/services/
```

---

**마이그레이션 완료 후에는 기존 파일을 삭제하지 말고 `_legacy` 접미사를 붙여 보관하세요!**
