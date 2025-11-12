# 프로젝트 데이터 흐름 및 통신 구조

## 📊 전체 아키텍처 개요

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│   데이터셋      │ ───> │   학습 파이프라인 │ ───> │  학습된 모델    │
│  (Food-101)     │      │  (notebooks/)     │      │ (best_model.pth)│
└─────────────────┘      └──────────────────┘      └─────────────────┘
                                                             │
                                                             │ 로드
                                                             ▼
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│  사용자 브라우저 │ <──> │  FastAPI 백엔드  │      │  사전학습 모델  │
│   (React App)   │ HTTP │  (api/main.py)   │ <──> │  (YOLO11n)      │
└─────────────────┘      └──────────────────┘      └─────────────────┘
```

---

## 1️⃣ 학습 파이프라인 (Training Pipeline)

### 1.1 데이터 수집
**경로**: `scripts/download_dataset.py`

```python
# 데이터 다운로드 흐름
Kaggle API → download_dataset.py → ~/.cache/kagglehub/
                                    │
                                    ├─ food-101/food-101/
                                    │  ├─ images/      # 101,000장 이미지
                                    │  └─ meta/        # 메타데이터
                                    │
                                    └─ dataset_path.txt (경로 저장)
```

**데이터 구조**:
```
food-101/
├── images/
│   ├── apple_pie/
│   │   ├── 1001.jpg
│   │   └── ...         (750장 train + 250장 test)
│   ├── baby_back_ribs/
│   └── ... (101개 클래스)
└── meta/
    ├── classes.txt     # 101개 클래스 이름
    ├── train.txt       # 훈련 이미지 목록 (75,750장)
    └── test.txt        # 테스트 이미지 목록 (25,250장)
```

### 1.2 데이터 전처리
**경로**: `notebooks/food101_training.py`

```python
# 데이터 로더 파이프라인
Raw Image (JPEG) → PIL.Image.open() → transforms.Compose([
                                        ├─ Resize(256)
                                        ├─ CenterCrop(224)
                                        ├─ ToTensor()
                                        └─ Normalize(mean, std)
                                      ]) → Tensor(3, 224, 224)
                                         → DataLoader(batch_size=128)
                                         → GPU Memory
```

**데이터 형식 변환**:
```
입력: JPEG 이미지 (다양한 크기)
  ↓
PIL Image (RGB, 다양한 크기)
  ↓
Resized (256 x 256)
  ↓
Center Cropped (224 x 224)
  ↓
Tensor (3, 224, 224), float32, [0, 1]
  ↓
Normalized Tensor (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ↓
Batch Tensor (128, 3, 224, 224) → GPU
```

### 1.3 모델 학습
**경로**: `notebooks/food101_training.py`

```python
# 학습 루프 데이터 흐름
Input Batch (128, 3, 224, 224)
  ↓
ResNet18 Forward Pass
  ├─ conv1, bn1, relu, maxpool
  ├─ layer1, layer2, layer3, layer4
  └─ avgpool, fc (512 → 101)
  ↓
Logits (128, 101)
  ↓
CrossEntropyLoss(logits, labels)
  ↓
Loss (scalar)
  ↓
Backward Pass (loss.backward())
  ↓
Optimizer.step() (Adam)
  ↓
Weight Update
```

**체크포인트 저장**:
```python
# outputs/models/best_model.pth 구조
{
    'epoch': 6,
    'model_state_dict': OrderedDict({
        'conv1.weight': Tensor(64, 3, 7, 7),
        'bn1.weight': Tensor(64),
        ...
        'fc.weight': Tensor(101, 512),
        'fc.bias': Tensor(101)
    }),
    'optimizer_state_dict': {...},
    'best_acc': 60.17
}
```

---

## 2️⃣ 백엔드 서버 (FastAPI)

### 2.1 서버 시작 시 초기화
**경로**: `api/main.py`

```python
# 서버 시작 시 데이터 흐름
@app.on_event("startup")
async def startup_event():
    1. 디바이스 설정 (CUDA/CPU)
    2. 클래스 로드 (data/food-101/meta/classes.txt)
       → CLASSES = ['apple_pie', 'baby_back_ribs', ...]
    3. 모델 초기화
       ResNet18 생성 → get_model(101)
    4. 가중치 로드
       torch.load('outputs/models/best_model.pth')
       → checkpoint['model_state_dict']
       → model.load_state_dict()
    5. GPU로 이동 (model.to('cuda'))
    6. 평가 모드 (model.eval())
```

### 2.2 API 엔드포인트별 데이터 흐름

#### A. `/predict` - 음식 분류

```
클라이언트 → HTTP POST /predict
   │
   └─ FormData: { file: <image_file> }
      │
      ▼
FastAPI 서버:
   1. UploadFile 수신
      await file.read() → bytes
      │
   2. 이미지 디코딩
      Image.open(BytesIO(bytes)) → PIL Image (RGB)
      │
   3. 전처리
      TRANSFORM(image) → Tensor(3, 224, 224)
      .unsqueeze(0) → Tensor(1, 3, 224, 224)
      .to(DEVICE) → GPU Tensor
      │
   4. 추론
      with torch.no_grad():
          outputs = MODEL(input_tensor)
          → Logits (1, 101)
      │
   5. 소프트맥스
      probabilities = torch.softmax(outputs, dim=1)[0]
      → Tensor(101) [0.001, 0.003, ..., 0.25, ...]
      │
   6. Top-5 추출
      top5_prob, top5_idx = torch.topk(probabilities, 5)
      │
   7. JSON 응답 생성
      {
        "success": true,
        "prediction": {
          "class": "apple_pie",
          "class_id": 0,
          "confidence": 0.25,
          "confidence_percent": "25.00%"
        },
        "top5": [...]
      }
      │
      ▼
클라이언트 ← HTTP 200 OK (JSON)
```

#### B. `/predict/gradcam` - Grad-CAM 히트맵

```
클라이언트 → HTTP POST /predict/gradcam
   │
   └─ FormData: { file: <image_file> }
      │
      ▼
FastAPI 서버:
   1-5. [/predict와 동일]
      │
   6. Grad-CAM 생성
      gradcam = GradCAM(model, target_layer=layer4[-1])
      │
      ├─ Forward Hook:
      │   activations = output.detach()  # (1, 512, 7, 7)
      │
      ├─ Forward Pass:
      │   outputs = model(input_tensor)
      │   → Logits (1, 101)
      │
      ├─ Backward Hook:
      │   model.zero_grad()
      │   class_score = outputs[0, target_class]
      │   class_score.backward()
      │   gradients = grad_output[0].detach()  # (1, 512, 7, 7)
      │
      ├─ CAM 계산:
      │   weights = gradients.mean(dim=[2, 3])  # (1, 512, 1, 1)
      │   cam = (weights * activations).sum(dim=1)  # (1, 7, 7)
      │   cam = F.relu(cam)
      │   cam = normalize(cam)  # [0, 1]
      │
      ├─ 리사이즈 & 컬러맵:
      │   cam_resized = cv2.resize(cam, (224, 224))
      │   heatmap = cv2.applyColorMap(cam_resized, COLORMAP_JET)
      │   overlay = heatmap * 0.4 + original_image * 0.6
      │
      └─ Base64 인코딩:
          buffered = BytesIO()
          overlay.save(buffered, format="PNG")
          base64.b64encode(buffered.getvalue())
      │
   7. JSON 응답
      {
        "success": true,
        "prediction": {...},
        "top5": [...],
        "gradcam": {
          "heatmap_image": "data:image/png;base64,iVBORw0...",
          "description": "빨간색 영역이 중요한 부분"
        }
      }
      │
      ▼
클라이언트 ← HTTP 200 OK (JSON + base64 이미지)
```

#### C. `/detect` - YOLO 객체 탐지

```
클라이언트 → HTTP POST /detect
   │
   └─ FormData: { file: <image_file> }
      │
      ▼
FastAPI 서버:
   1. 이미지 로드
      Image.open(BytesIO(bytes)) → PIL Image (RGB)
      │
   2. PIL → numpy 변환
      np.array(image) → ndarray(H, W, 3)
      │
   3. YOLO 예측
      detector = YOLODetector('yolo11n.pt')
      results = model.predict(
          source=image_array,
          conf=0.25,
          verbose=False
      )
      │
      ├─ 전처리 (YOLO 내부):
      │   Letterbox Resize → (640, 640)
      │   Normalize → [0, 1]
      │   Tensor 변환
      │
      ├─ 추론:
      │   YOLO11n Forward
      │   → Detections [(x1,y1,x2,y2,conf,cls), ...]
      │
      └─ NMS (Non-Maximum Suppression):
          중복 박스 제거
      │
   4. 탐지 결과 추출
      boxes = result.boxes.xyxy  # (N, 4)
      confidences = result.boxes.conf  # (N,)
      class_ids = result.boxes.cls  # (N,)
      │
   5. 어노테이션 이미지 생성
      annotated = result.plot()  # BGR numpy array
      → RGB 변환 → PIL Image
      → Base64 인코딩
      │
   6. JSON 응답
      {
        "success": true,
        "num_objects": 3,
        "detections": [
          {
            "class": "person",
            "confidence": 0.85,
            "bbox": {
              "x1": 100, "y1": 150,
              "x2": 300, "y2": 500,
              "width": 200, "height": 350
            }
          },
          ...
        ],
        "annotated_image": "data:image/png;base64,..."
      }
      │
      ▼
클라이언트 ← HTTP 200 OK (JSON + base64 이미지)
```

---

## 3️⃣ 프론트엔드 (React)

### 3.1 컴포넌트 상태 관리
**경로**: `frontend/src/App.js`

```javascript
// React State (메모리)
const [image, setImage] = useState(null);           // File 객체
const [preview, setPreview] = useState(null);       // Data URL (base64)
const [result, setResult] = useState(null);         // API 응답 JSON
const [loading, setLoading] = useState(false);      // 로딩 상태
const [mode, setMode] = useState('classify');       // 'classify' | 'detect'
const [showGradCAM, setShowGradCAM] = useState(false);  // boolean
```

### 3.2 이미지 업로드 흐름

```
사용자 액션 (드래그 & 드롭 / 파일 선택 / 붙여넣기)
   │
   ▼
handleFileSelect(file):
   1. File 객체 검증
      file.type.startsWith('image/')
      │
   2. State 업데이트
      setImage(file)  // File 객체 저장
      │
   3. 미리보기 생성
      FileReader.readAsDataURL(file)
      │
      ├─ onload: (e) => {
      │     setPreview(e.target.result)
      │     // "data:image/jpeg;base64,/9j/4AAQ..."
      │   }
      │
      └─ 브라우저 메모리에 저장
   │
   ▼
화면 렌더링:
   <img src={preview} />  // Data URL로 이미지 표시
```

### 3.3 API 요청 흐름

```javascript
// handlePredict() 함수
사용자가 "분석하기" 클릭
   │
   ▼
1. 엔드포인트 결정
   if (mode === 'detect') {
       endpoint = '/detect'
   } else if (showGradCAM) {
       endpoint = '/predict/gradcam'
   } else {
       endpoint = '/predict'
   }
   │
   ▼
2. FormData 생성
   const formData = new FormData()
   formData.append('file', image)  // File 객체
   │
   │ FormData 구조:
   │ ┌─────────────────────────────────┐
   │ │ Content-Type: multipart/form-data│
   │ │ boundary: ----WebKitFormBoundary │
   │ │                                   │
   │ │ ----WebKitFormBoundary           │
   │ │ Content-Disposition: form-data;  │
   │ │   name="file"; filename="img.jpg"│
   │ │ Content-Type: image/jpeg         │
   │ │                                   │
   │ │ <binary image data>              │
   │ │ ----WebKitFormBoundary--         │
   │ └─────────────────────────────────┘
   │
   ▼
3. axios POST 요청
   axios.post(`${API_URL}${endpoint}`, formData, {
     headers: { 'Content-Type': 'multipart/form-data' }
   })
   │
   │ HTTP 요청:
   │ POST http://localhost:8000/predict
   │ Content-Type: multipart/form-data; boundary=...
   │ Content-Length: <size>
   │
   │ <FormData body>
   │
   ▼
4. 네트워크 전송
   Browser → TCP/IP → FastAPI Server (localhost:8000)
   │
   ▼
5. 서버 응답 대기
   [FastAPI 처리... 2-3초]
   │
   ▼
6. 응답 수신
   HTTP/1.1 200 OK
   Content-Type: application/json

   {
     "success": true,
     "prediction": {...},
     "top5": [...],
     "gradcam": {
       "heatmap_image": "data:image/png;base64,iVBORw0KGgo..."
     }
   }
   │
   ▼
7. State 업데이트
   setResult(response.data)
   │
   ▼
8. 화면 렌더링
   {result && (
     <div>
       <img src={result.gradcam.heatmap_image} />
       <p>{result.prediction.class}</p>
       <p>{result.prediction.confidence_percent}</p>
     </div>
   )}
```

### 3.4 이미지 렌더링 방식

```javascript
// 세 가지 이미지 렌더링 방식

1. 미리보기 (로컬 파일)
   <img src={preview} />
   // preview = "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
   // 브라우저가 Data URL을 디코딩하여 표시

2. Grad-CAM 히트맵 (서버 응답)
   <img src={result.gradcam.heatmap_image} />
   // "data:image/png;base64,iVBORw0KGgoAAAANSUh..."
   // 서버에서 생성한 PNG를 base64로 인코딩
   // 브라우저가 디코딩하여 표시

3. YOLO 탐지 결과 (서버 응답)
   <img src={result.annotated_image} />
   // "data:image/png;base64,iVBORw0KGgoAAAANSUh..."
   // YOLO가 바운딩 박스를 그린 이미지
```

---

## 4️⃣ 통신 프로토콜 상세

### 4.1 HTTP 요청/응답 형식

#### 요청 (Request)
```http
POST /predict HTTP/1.1
Host: localhost:8000
Content-Type: multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Length: 245687

------WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="file"; filename="pizza.jpg"
Content-Type: image/jpeg

[BINARY IMAGE DATA - 245,687 bytes]
------WebKitFormBoundary7MA4YWxkTrZu0gW--
```

#### 응답 (Response)
```http
HTTP/1.1 200 OK
Content-Type: application/json
Content-Length: 1234
Access-Control-Allow-Origin: *

{
  "success": true,
  "prediction": {
    "class": "pizza",
    "class_id": 53,
    "confidence": 0.8523,
    "confidence_percent": "85.23%"
  },
  "top5": [
    {
      "rank": 1,
      "class": "pizza",
      "class_id": 53,
      "confidence": 0.8523,
      "confidence_percent": "85.23%"
    },
    ...
  ]
}
```

### 4.2 CORS (Cross-Origin Resource Sharing)

```python
# api/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # 모든 origin 허용
    allow_credentials=True,
    allow_methods=["*"],           # GET, POST, PUT, DELETE 등
    allow_headers=["*"],           # 모든 헤더 허용
)
```

**CORS 흐름**:
```
1. Preflight Request (OPTIONS)
   Browser → Server:
   OPTIONS /predict HTTP/1.1
   Origin: http://localhost:3000
   Access-Control-Request-Method: POST

   Server → Browser:
   HTTP/1.1 200 OK
   Access-Control-Allow-Origin: *
   Access-Control-Allow-Methods: POST, GET, OPTIONS
   Access-Control-Allow-Headers: Content-Type

2. Actual Request (POST)
   Browser → Server:
   POST /predict HTTP/1.1
   Origin: http://localhost:3000

   Server → Browser:
   HTTP/1.1 200 OK
   Access-Control-Allow-Origin: *
   { "success": true, ... }
```

---

## 5️⃣ 데이터 형식 변환 체인

### 음식 분류 전체 흐름

```
📷 사용자 이미지 (pizza.jpg, 1.2MB)
    │
    ├─ 브라우저 파일 시스템
    │  → File 객체
    │
    ├─ FileReader.readAsDataURL()
    │  → Data URL (base64)
    │  → "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
    │
    ├─ FormData + HTTP POST
    │  → multipart/form-data (바이너리)
    │  → 네트워크 전송
    │
    ├─ FastAPI UploadFile
    │  → await file.read()
    │  → bytes (바이너리)
    │
    ├─ PIL Image.open()
    │  → PIL Image (RGB, 800x600)
    │
    ├─ transforms.Compose()
    │  → Tensor(3, 224, 224), dtype=float32, range=[0,1]
    │
    ├─ Normalize()
    │  → Tensor(3, 224, 224), mean=[0.485,...], std=[0.229,...]
    │
    ├─ .unsqueeze(0)
    │  → Tensor(1, 3, 224, 224)
    │
    ├─ .to('cuda')
    │  → GPU Tensor(1, 3, 224, 224)
    │
    ├─ ResNet18 Forward
    │  → Logits Tensor(1, 101)
    │
    ├─ torch.softmax()
    │  → Probabilities Tensor(101), sum=1.0
    │
    ├─ .cpu().item()
    │  → Python float (0.8523)
    │
    ├─ JSON 직렬화
    │  → {"confidence": 0.8523, ...}
    │
    ├─ HTTP Response
    │  → Content-Type: application/json
    │
    ├─ axios 파싱
    │  → JavaScript 객체
    │
    └─ React State
       → setResult({ prediction: { confidence: 0.8523 } })
       → 화면 렌더링
```

---

## 6️⃣ 메모리 및 저장소 위치

### 학습 시
```
RAM (CPU):
  - Python 프로그램 (100MB)
  - DataLoader 버퍼 (4 workers × 2 batches × 128 × 3 × 224 × 224 × 4 bytes ≈ 1.5GB)

GPU Memory:
  - 모델 파라미터 (ResNet18: ~45MB)
  - Optimizer State (Adam: ~90MB)
  - Forward Activations (배치당 ~200MB)
  - Gradients (~45MB)
  - 총: ~380MB

Disk:
  - 데이터셋: /root/.cache/kagglehub/ (11GB)
  - 체크포인트: /root/llm_prj/outputs/models/best_model.pth (129MB)
```

### 서비스 시
```
서버 RAM:
  - FastAPI 프로세스 (~200MB)
  - 모델 파라미터 (CPU에도 복사, ~45MB)

서버 GPU:
  - 모델 파라미터 (~45MB)
  - 입력 텐서 (배치당 ~1MB)
  - Forward Activations (~10MB)
  - 총: ~60MB

클라이언트 (브라우저):
  - React 앱 (~10MB)
  - 이미지 미리보기 (Data URL, ~1-3MB)
  - API 응답 캐시 (~500KB)
```

---

## 7️⃣ 성능 최적화 포인트

### 데이터 로딩
```python
# DataLoader 최적화
DataLoader(
    num_workers=4,           # 4개 프로세스로 병렬 로딩
    pin_memory=False,        # GPU 메모리 절약
    persistent_workers=True, # 워커 재사용
    prefetch_factor=2,       # 미리 2배치 준비
    drop_last=True           # GPU 효율성
)
```

### 추론 최적화
```python
# Mixed Precision (자동 혼합 정밀도)
with autocast(device_type='cuda'):
    outputs = model(inputs)
    # float16으로 계산 → 2배 빠름, 메모리 절약

# Gradient 비활성화
with torch.no_grad():
    outputs = model(inputs)
    # 메모리 절약, 속도 향상

# 배치 처리
# 단일 이미지: ~50ms
# 배치 128: ~500ms (이미지당 4ms)
```

### 네트워크 최적화
```javascript
// React - axios 요청
- Content-Type: multipart/form-data (효율적인 바이너리 전송)
- 이미지 압축: JPEG quality 조절
- 응답 캐싱: React State로 결과 저장

// FastAPI
- async/await: 비동기 처리
- 스트리밍 응답: 대용량 파일
```

---

## 8️⃣ 에러 처리 및 검증

### 클라이언트 검증
```javascript
// frontend/src/App.js
if (!file.type.startsWith('image/')) {
  setError('이미지 파일만 업로드 가능합니다.')
  return
}
```

### 서버 검증
```python
# api/main.py
try:
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
except Exception as e:
    raise HTTPException(
        status_code=400,
        detail="유효하지 않은 이미지 파일입니다"
    )
```

### 에러 응답 형식
```json
{
  "detail": "예측 중 오류 발생: Invalid image format"
}
```

---

## 📝 요약

### 데이터 흐름 3단계

1. **학습 단계** (오프라인)
   ```
   Raw 데이터 → 전처리 → 모델 학습 → 체크포인트 저장
   ```

2. **서버 시작** (1회)
   ```
   체크포인트 로드 → GPU 메모리 → 서비스 대기
   ```

3. **추론 요청** (실시간)
   ```
   이미지 업로드 → 전처리 → 모델 추론 → JSON 응답 → 화면 표시
   ```

### 주요 통신 방식

- **학습 ↔ 디스크**: PyTorch save/load (pickle)
- **클라이언트 ↔ 서버**: HTTP/JSON (REST API)
- **서버 ↔ 모델**: Python 함수 호출 (in-memory)
- **서버 ↔ GPU**: CUDA 메모리 전송

이 구조는 확장 가능하며, 각 컴포넌트가 독립적으로 동작하여 유지보수가 용이합니다.
