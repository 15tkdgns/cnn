"""
Food-101 Image Classification API
FastAPI를 사용한 음식 이미지 분류 서버

주요 기능:
- ResNet18 기반 음식 이미지 분류 (101개 클래스)
- Grad-CAM 히트맵 시각화 (AI 판단 근거 표시)
- YOLO 객체 탐지 (80개 클래스)
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import io
from pathlib import Path
from typing import List, Dict
import logging
import base64

# Grad-CAM import
# 패키지 실행과 직접 실행 모두 지원하기 위한 이중 import
try:
    from .gradcam import create_gradcam_visualization
except ImportError:
    from gradcam import create_gradcam_visualization

# YOLO import
# 패키지 실행과 직접 실행 모두 지원하기 위한 이중 import
try:
    from .yolo_detector import get_yolo_detector
except ImportError:
    from yolo_detector import get_yolo_detector

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="Food-101 Image Classification API",
    description="ResNet18 기반 음식 이미지 분류 API",
    version="1.0.0"
)

# CORS 설정 (Cross-Origin Resource Sharing)
# 다른 도메인(localhost:3000)에서 실행되는 프론트엔드가 이 API에 접근할 수 있도록 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 환경: 모든 origin 허용, 운영 환경에서는 특정 도메인만 허용 권장
    allow_credentials=True,  # 쿠키를 포함한 인증 정보 허용
    allow_methods=["*"],  # 모든 HTTP 메서드 허용 (GET, POST, PUT, DELETE 등)
    allow_headers=["*"],  # 모든 HTTP 헤더 허용
)

# 전역 변수
# 서버 시작 시 한 번만 로드하여 매 요청마다 다시 로드하는 비효율 방지
MODEL = None        # ResNet18 모델 인스턴스
DEVICE = None       # 연산 디바이스 (cuda 또는 cpu)
CLASSES = []        # 101개 음식 클래스 이름 리스트
TRANSFORM = None    # 이미지 전처리 파이프라인


def load_classes():
    """
    Food-101 데이터셋의 클래스 이름 목록을 로드

    Returns:
        list: 101개 음식 클래스 이름 (예: ['apple_pie', 'baby_back_ribs', ...])

    동작 과정:
    1. dataset_path.txt에서 데이터셋 경로 읽기 (있는 경우)
    2. meta/classes.txt 파일에서 클래스 이름 로드
    3. 파일이 없으면 기본 클래스 목록 반환 (class_0, class_1, ...)
    """
    try:
        # 기본 경로: 프로젝트 루트/data/food-101/...
        meta_dir = Path(__file__).parent.parent / "data" / "food-101" / "food-101" / "meta"

        # dataset_path.txt가 있으면 그 경로를 우선 사용
        # (Kaggle 다운로드 시 ~/.cache/kagglehub/에 저장되므로)
        dataset_path_file = Path(__file__).parent.parent / "data" / "dataset_path.txt"
        if dataset_path_file.exists():
            with open(dataset_path_file, 'r') as f:
                base_path = Path(f.read().strip())
            meta_dir = base_path / "food-101" / "food-101" / "meta"

        classes_file = meta_dir / "classes.txt"

        # classes.txt 파일이 없으면 기본 클래스 이름 사용
        if not classes_file.exists():
            logger.warning(f"클래스 파일을 찾을 수 없습니다: {classes_file}")
            logger.warning("기본 클래스 목록을 사용합니다 (class_0 ~ class_100)")
            return [f"class_{i}" for i in range(101)]

        # 클래스 파일 읽기 (한 줄에 하나씩 클래스 이름)
        with open(classes_file, 'r') as f:
            classes = [line.strip() for line in f]

        logger.info(f"[SUCCESS] {len(classes)}개 클래스 로드 완료")
        return classes

    except Exception as e:
        logger.error(f"클래스 로드 실패: {e}")
        # 예외 발생 시 기본 클래스 목록 반환
        return [f"class_{i}" for i in range(101)]


def get_model(num_classes: int = 101):
    """
    ResNet18 모델 구조 생성 (Transfer Learning용)

    Args:
        num_classes: 출력 클래스 개수 (기본값: 101)

    Returns:
        nn.Module: Food-101 분류용으로 수정된 ResNet18 모델

    동작 과정:
    1. ImageNet으로 사전학습된 ResNet18 구조 로드 (가중치는 미포함)
    2. 마지막 Fully Connected Layer를 101개 클래스 출력으로 교체
       - 원본: fc(512 -> 1000) [ImageNet 1000 클래스]
       - 변경: fc(512 -> 101)   [Food-101 101 클래스]
    """
    model = models.resnet18(weights=None)  # 구조만 로드, 가중치는 checkpoint에서 로드
    num_features = model.fc.in_features    # ResNet18의 경우 512
    model.fc = nn.Linear(num_features, num_classes)  # 출력층 교체
    return model


def load_model():
    """
    훈련된 ResNet18 모델과 관련 설정을 메모리에 로드

    서버 시작 시 한 번만 실행되어 전역 변수 초기화:
    - MODEL: 학습된 ResNet18 모델
    - DEVICE: GPU 또는 CPU 디바이스
    - CLASSES: 101개 음식 클래스 이름
    - TRANSFORM: 이미지 전처리 파이프라인

    동작 과정:
    1. GPU 사용 가능 여부 확인 후 DEVICE 설정
    2. classes.txt에서 클래스 이름 로드
    3. ResNet18 모델 구조 생성 (fc layer는 101 출력)
    4. best_model.pth에서 학습된 가중치 로드
    5. 모델을 DEVICE로 이동 및 평가 모드 설정
    6. 이미지 전처리 파이프라인 구성
    """
    global MODEL, DEVICE, CLASSES, TRANSFORM

    try:
        # Step 1: 연산 디바이스 설정 (GPU 사용 가능하면 cuda, 아니면 cpu)
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"[DEVICE] 연산 디바이스: {DEVICE}")

        # Step 2: 클래스 이름 로드 (apple_pie, baby_back_ribs, ...)
        CLASSES = load_classes()

        # Step 3: 모델 구조 생성 (가중치는 아직 미포함)
        MODEL = get_model(len(CLASSES))

        # Step 4: 학습된 모델 가중치 로드
        model_path = Path(__file__).parent.parent / "outputs" / "models" / "best_model.pth"

        if not model_path.exists():
            # 모델 파일이 없으면 경고 메시지 출력 (초기화된 가중치로 실행)
            logger.warning(f"[WARNING] 모델 파일을 찾을 수 없습니다: {model_path}")
            logger.warning("[WARNING] 사전학습된 가중치 없이 실행됩니다 (정확도가 낮을 수 있습니다)")
        else:
            # 체크포인트 로드 (CPU로 먼저 로드 후 DEVICE로 이동)
            checkpoint = torch.load(model_path, map_location=DEVICE)

            # 체크포인트 형식 확인
            # 형식 1: {'model_state_dict': ..., 'epoch': ..., 'best_acc': ...}
            # 형식 2: state_dict 직접 저장
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                epoch = checkpoint.get('epoch', 'N/A')
                best_acc = checkpoint.get('best_acc', 'N/A')
                logger.info(f"[SUCCESS] 체크포인트 로드 (Epoch: {epoch}, Best Acc: {best_acc:.2f}%)")
            else:
                state_dict = checkpoint

            # 모델에 가중치 적용
            MODEL.load_state_dict(state_dict)
            logger.info(f"[SUCCESS] 모델 로드 완료: {model_path}")

        # Step 5: 모델을 선택한 디바이스로 이동 및 평가 모드 설정
        MODEL = MODEL.to(DEVICE)
        MODEL.eval()  # Dropout, BatchNorm을 평가 모드로 전환

        # Step 6: 이미지 전처리 파이프라인 구성
        # 훈련 시와 동일한 전처리를 적용해야 정확한 예측 가능
        TRANSFORM = transforms.Compose([
            transforms.Resize(256),           # 짧은 변을 256픽셀로 리사이즈
            transforms.CenterCrop(224),       # 중앙 224x224 크롭
            transforms.ToTensor(),            # PIL Image -> Tensor (0~1 범위)
            transforms.Normalize(             # ImageNet 평균/표준편차로 정규화
                mean=[0.485, 0.456, 0.406],   # RGB 채널별 평균
                std=[0.229, 0.224, 0.225]     # RGB 채널별 표준편차
            )
        ])

        logger.info("[SUCCESS] 모델 초기화 완료")

    except Exception as e:
        logger.error(f"[ERROR] 모델 로드 실패: {e}")
        raise


# Static 파일 마운트 (HTML 인터페이스)
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


@app.on_event("startup")
async def startup_event():
    """
    FastAPI 서버 시작 이벤트 핸들러

    서버가 시작될 때 자동으로 실행되어 모델을 메모리에 로드합니다.
    이렇게 하면 첫 요청이 들어왔을 때 모델 로딩으로 인한 지연이 없습니다.
    """
    logger.info("[STARTUP] Food-101 API 서버 시작 중...")
    load_model()
    logger.info("[STARTUP] 서버 준비 완료!")


@app.get("/")
async def root():
    """루트 엔드포인트 - 웹 인터페이스 제공"""
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        return FileResponse(str(html_path))
    else:
        return {
            "message": "Food-101 Image Classification API",
            "version": "1.0.0",
            "endpoints": {
                "predict": "POST /predict - 이미지 업로드 및 분류",
                "health": "GET /health - 서버 상태 확인",
                "classes": "GET /classes - 분류 가능한 음식 목록",
                "docs": "GET /docs - API 문서 (Swagger UI)",
                "web": "GET / - 웹 인터페이스"
            }
        }


@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    is_healthy = MODEL is not None and DEVICE is not None

    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "model_loaded": MODEL is not None,
        "device": str(DEVICE) if DEVICE else None,
        "num_classes": len(CLASSES)
    }


@app.get("/classes")
async def get_classes():
    """분류 가능한 음식 클래스 목록 반환"""
    return {
        "total": len(CLASSES),
        "classes": CLASSES
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    이미지를 업로드하여 음식 종류 예측 (ResNet18 기반)

    Args:
        file: 업로드된 이미지 파일 (JPEG, PNG 등)

    Returns:
        dict: {
            "success": True,
            "prediction": {
                "class": "pizza",           # 예측된 음식 클래스 이름
                "class_id": 72,             # 클래스 ID (0~100)
                "confidence": 0.952,        # 신뢰도 (0~1)
                "confidence_percent": "95.20%"
            },
            "top5": [                       # 상위 5개 예측 결과
                {"rank": 1, "class": "pizza", "confidence": 0.952, ...},
                {"rank": 2, "class": "lasagna", "confidence": 0.023, ...},
                ...
            ]
        }

    동작 과정:
    1. 업로드된 이미지를 PIL Image로 변환 (RGB 모드)
    2. 전처리 파이프라인 적용 (Resize -> CenterCrop -> Normalize)
    3. Tensor를 GPU/CPU로 이동
    4. 모델에 입력하여 예측 (101개 클래스에 대한 확률 계산)
    5. Top-5 예측 결과 반환
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다")

    try:
        # Step 1: 이미지 파일 읽기 및 RGB 변환
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Step 2: 이미지 전처리
        # PIL Image (H, W, 3) -> Tensor (3, 224, 224) -> 정규화
        input_tensor = TRANSFORM(image).unsqueeze(0)  # (1, 3, 224, 224) 배치 차원 추가
        input_tensor = input_tensor.to(DEVICE)        # GPU/CPU로 이동

        # Step 3: 예측 수행
        with torch.no_grad():  # 그래디언트 계산 비활성화 (추론 시 불필요)
            outputs = MODEL(input_tensor)  # (1, 101) logits
            probabilities = torch.softmax(outputs, dim=1)[0]  # (101,) 확률값으로 변환

            # Top-5 예측 추출
            top5_prob, top5_idx = torch.topk(probabilities, 5)

            # 최고 예측 (Top-1)
            top1_idx = top5_idx[0].item()
            top1_prob = top5_prob[0].item()

        # Step 4: 결과 반환
        return {
            "success": True,
            "prediction": {
                "class": CLASSES[top1_idx],
                "class_id": top1_idx,
                "confidence": float(top1_prob),
                "confidence_percent": f"{top1_prob * 100:.2f}%"
            },
            "top5": [
                {
                    "rank": i + 1,
                    "class": CLASSES[idx.item()],
                    "class_id": idx.item(),
                    "confidence": float(prob.item()),
                    "confidence_percent": f"{prob.item() * 100:.2f}%"
                }
                for i, (prob, idx) in enumerate(zip(top5_prob, top5_idx))
            ]
        }

    except Exception as e:
        logger.error(f"예측 실패: {e}")
        raise HTTPException(status_code=500, detail=f"예측 중 오류 발생: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    여러 이미지를 한 번에 배치 예측 (최대 10개)

    Args:
        files: 업로드된 이미지 파일 리스트

    Returns:
        dict: {
            "success": True,
            "total": 3,
            "results": [
                {"filename": "image1.jpg", "prediction": {...}},
                {"filename": "image2.jpg", "prediction": {...}},
                {"filename": "image3.jpg", "error": "..."}  # 실패한 경우
            ]
        }

    동작 과정:
    1. 파일 개수 제한 확인 (최대 10개)
    2. 각 이미지를 순차적으로 처리
    3. 각 이미지마다 Top-1 예측 수행
    4. 성공/실패 결과를 리스트로 반환
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다")

    # 배치 크기 제한 (너무 많으면 메모리 부족 가능)
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="한 번에 최대 10개 이미지만 처리 가능합니다")

    results = []

    # 각 이미지를 순차적으로 처리
    for file in files:
        try:
            # 이미지 읽기 및 전처리
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

            input_tensor = TRANSFORM(image).unsqueeze(0)
            input_tensor = input_tensor.to(DEVICE)

            # 예측 수행 (Top-1만 반환)
            with torch.no_grad():
                outputs = MODEL(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
                top1_prob, top1_idx = torch.max(probabilities, dim=0)

            # 성공 결과 추가
            results.append({
                "filename": file.filename,
                "prediction": {
                    "class": CLASSES[top1_idx.item()],
                    "class_id": top1_idx.item(),
                    "confidence": float(top1_prob.item()),
                    "confidence_percent": f"{top1_prob.item() * 100:.2f}%"
                }
            })

        except Exception as e:
            # 실패한 이미지는 에러 메시지 추가
            results.append({
                "filename": file.filename,
                "error": str(e)
            })

    return {
        "success": True,
        "total": len(files),
        "results": results
    }


@app.post("/predict/gradcam")
async def predict_with_gradcam(file: UploadFile = File(...)):
    """
    이미지를 업로드하여 음식 종류 예측 및 Grad-CAM 히트맵 생성

    Grad-CAM (Gradient-weighted Class Activation Mapping):
    - AI가 어느 부분을 보고 판단했는지 시각화
    - 빨간색/노란색: 중요하게 본 부분 (높은 활성도)
    - 파란색/보라색: 덜 중요한 부분 (낮은 활성도)

    Args:
        file: 업로드된 이미지 파일 (JPEG, PNG 등)

    Returns:
        dict: {
            "success": True,
            "prediction": {...},           # 일반 예측 결과
            "top5": [...],                 # Top-5 예측
            "gradcam": {
                "heatmap_image": "data:image/png;base64,...",  # 히트맵 오버레이 이미지
                "description": "빨간색 영역이 모델이 판단할 때 중요하게 본 부분입니다"
            }
        }

    동작 과정:
    1. 이미지 전처리 및 일반 예측 수행
    2. Grad-CAM으로 중간층 활성도 맵 생성
    3. 원본 이미지에 히트맵 오버레이
    4. 결과를 base64로 인코딩하여 반환
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다")

    try:
        # Step 1: 이미지 파일 읽기
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Step 2: 이미지 전처리
        input_tensor = TRANSFORM(image).unsqueeze(0)  # (1, 3, 224, 224)
        input_tensor = input_tensor.to(DEVICE)

        # Step 3: 예측 수행 (일반 분류)
        with torch.no_grad():
            outputs = MODEL(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]

            # Top-5 예측
            top5_prob, top5_idx = torch.topk(probabilities, 5)

            # 최고 예측
            top1_idx = top5_idx[0].item()
            top1_prob = top5_prob[0].item()

        # Step 4: Grad-CAM 히트맵 생성
        # ResNet18의 layer4 활성도를 기반으로 히트맵 생성
        overlay_image, pred_class, pred_prob = create_gradcam_visualization(
            MODEL, input_tensor, image, DEVICE
        )

        # Step 5: 히트맵 이미지를 base64로 인코딩
        # 프론트엔드에서 <img src="data:image/png;base64,..."> 형태로 사용
        buffered = io.BytesIO()
        overlay_image.save(buffered, format="PNG")
        heatmap_base64 = base64.b64encode(buffered.getvalue()).decode()

        # Step 6: 결과 반환
        return {
            "success": True,
            "prediction": {
                "class": CLASSES[top1_idx],
                "class_id": top1_idx,
                "confidence": float(top1_prob),
                "confidence_percent": f"{top1_prob * 100:.2f}%"
            },
            "top5": [
                {
                    "rank": i + 1,
                    "class": CLASSES[idx.item()],
                    "class_id": idx.item(),
                    "confidence": float(prob.item()),
                    "confidence_percent": f"{prob.item() * 100:.2f}%"
                }
                for i, (prob, idx) in enumerate(zip(top5_prob, top5_idx))
            ],
            "gradcam": {
                "heatmap_image": f"data:image/png;base64,{heatmap_base64}",
                "description": "빨간색 영역이 모델이 판단할 때 중요하게 본 부분입니다"
            }
        }

    except Exception as e:
        logger.error(f"Grad-CAM 예측 실패: {e}")
        raise HTTPException(status_code=500, detail=f"예측 중 오류 발생: {str(e)}")


@app.post("/detect")
async def detect_objects(file: UploadFile = File(...), conf_threshold: float = 0.25):
    """
    YOLO11n을 사용한 일반 객체 탐지 (80개 클래스)

    YOLO (You Only Look Once):
    - 실시간 객체 탐지 모델
    - COCO 데이터셋으로 학습 (사람, 차, 동물 등 80개 클래스)
    - 바운딩 박스와 클래스, 신뢰도 반환

    Args:
        file: 업로드된 이미지 파일 (JPEG, PNG 등)
        conf_threshold: 신뢰도 임계값 (0.0 ~ 1.0, 기본: 0.25)
                       이 값보다 높은 신뢰도의 객체만 탐지

    Returns:
        dict: {
            "success": True,
            "num_objects": 3,
            "detections": [
                {
                    "class": "person",
                    "class_id": 0,
                    "confidence": 0.89,
                    "confidence_percent": "89.00%",
                    "bbox": {
                        "x1": 100, "y1": 50,      # 좌상단 좌표
                        "x2": 300, "y2": 400,     # 우하단 좌표
                        "width": 200, "height": 350
                    }
                },
                ...
            ],
            "annotated_image": "data:image/png;base64,..."  # 바운딩 박스가 그려진 이미지
        }

    동작 과정:
    1. 이미지를 PIL Image로 변환
    2. YOLO 모델로 객체 탐지 수행
    3. 신뢰도가 임계값 이상인 객체만 필터링
    4. 바운딩 박스가 그려진 이미지를 base64로 인코딩
    """
    try:
        # Step 1: 이미지 파일 읽기
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Step 2: YOLO 탐지기 인스턴스 가져오기 (싱글톤 패턴)
        detector = get_yolo_detector()

        # Step 3: 객체 탐지 수행
        # - 이미지에서 객체 탐지
        # - 바운딩 박스가 그려진 이미지 생성
        # - 결과를 base64로 인코딩
        result = detector.detect_and_encode(image, conf_threshold)

        # Step 4: 결과 반환
        return {
            "success": True,
            "num_objects": result['num_objects'],
            "detections": result['detections'],
            "annotated_image": result['annotated_image_base64']
        }

    except Exception as e:
        logger.error(f"YOLO 객체 탐지 실패: {e}")
        raise HTTPException(status_code=500, detail=f"객체 탐지 중 오류 발생: {str(e)}")


@app.get("/detect/classes")
async def get_yolo_classes():
    """YOLO 모델이 탐지할 수 있는 클래스 목록 반환"""
    try:
        detector = get_yolo_detector()
        class_names = detector.get_class_names()

        return {
            "success": True,
            "total_classes": len(class_names),
            "classes": class_names
        }

    except Exception as e:
        logger.error(f"YOLO 클래스 목록 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"클래스 목록 조회 중 오류 발생: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
