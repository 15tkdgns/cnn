"""
Food-101 Image Classification API (리팩토링 버전)
FastAPI를 사용한 음식 이미지 분류 서버

주요 기능:
- ResNet18 기반 음식 이미지 분류 (101개 클래스)
- Grad-CAM 히트맵 시각화 (AI 판단 근거 표시)
- YOLO 객체 탐지 (80개 클래스)

개선 사항:
- 설정 중앙화 (config.py)
- Pydantic 모델로 타입 안전성 보장
- 유틸리티 함수 분리
- 로깅 개선
- 보안 강화 (파일 검증)
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path
from typing import List
import base64
import io
from contextlib import asynccontextmanager
from time import time

# 로컬 모듈 임포트
from .config import settings
from .models import (
    ClassificationResponse,
    DetectionResponse,
    HealthCheckResponse,
    ClassesResponse,
    PredictionResult,
    Top5Prediction,
    GradCAMResult,
    Detection,
    BoundingBox
)
from .utils import (
    load_classes,
    validate_and_load_image,
    format_confidence
)
from .logger import api_logger, log_request, log_prediction

# Grad-CAM 및 YOLO import
try:
    from .gradcam import create_gradcam_visualization
    from .yolo_detector import get_yolo_detector
except ImportError:
    from gradcam import create_gradcam_visualization
    from yolo_detector import get_yolo_detector


# =============================================================================
# 전역 변수
# =============================================================================

MODEL = None        # ResNet18 모델 인스턴스
DEVICE = None       # 연산 디바이스 (cuda 또는 cpu)
CLASSES = []        # 101개 음식 클래스 이름 리스트
TRANSFORM = None    # 이미지 전처리 파이프라인


# =============================================================================
# 모델 관리 함수
# =============================================================================

def get_model(num_classes: int = 101) -> nn.Module:
    """
    ResNet18 모델 구조 생성 (Transfer Learning용)

    Args:
        num_classes: 출력 클래스 개수 (기본값: 101)

    Returns:
        nn.Module: Food-101 분류용으로 수정된 ResNet18 모델
    """
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model


def load_model():
    """
    훈련된 ResNet18 모델과 관련 설정을 메모리에 로드

    전역 변수 초기화:
    - MODEL: 학습된 ResNet18 모델
    - DEVICE: GPU 또는 CPU 디바이스
    - CLASSES: 101개 음식 클래스 이름
    - TRANSFORM: 이미지 전처리 파이프라인
    """
    global MODEL, DEVICE, CLASSES, TRANSFORM

    try:
        # Step 1: 연산 디바이스 설정
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        api_logger.info(f"[DEVICE] 연산 디바이스: {DEVICE}")

        # Step 2: 클래스 이름 로드
        CLASSES = load_classes(
            Path(settings.classes_file_path),
            Path(settings.dataset_path_file)
        )

        # Step 3: 모델 구조 생성
        MODEL = get_model(len(CLASSES))

        # Step 4: 학습된 모델 가중치 로드
        model_path = Path(settings.model_path)

        if not model_path.exists():
            api_logger.warning(f"[WARNING] 모델 파일을 찾을 수 없습니다: {model_path}")
            api_logger.warning("[WARNING] 사전학습된 가중치 없이 실행됩니다")
        else:
            checkpoint = torch.load(model_path, map_location=DEVICE)

            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                epoch = checkpoint.get('epoch', 'N/A')
                best_acc = checkpoint.get('best_acc', 'N/A')
                api_logger.info(f"[SUCCESS] 체크포인트 로드 (Epoch: {epoch}, Best Acc: {best_acc:.2f}%)")
            else:
                state_dict = checkpoint

            MODEL.load_state_dict(state_dict)
            api_logger.info(f"[SUCCESS] 모델 로드 완료: {model_path}")

        # Step 5: 모델을 디바이스로 이동 및 평가 모드 설정
        MODEL = MODEL.to(DEVICE)
        MODEL.eval()

        # Step 6: 이미지 전처리 파이프라인 구성
        TRANSFORM = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        api_logger.info("[SUCCESS] 모델 초기화 완료")

    except Exception as e:
        api_logger.error(f"[ERROR] 모델 로드 실패: {e}", exc_info=True)
        raise


# =============================================================================
# FastAPI 애플리케이션 설정
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    애플리케이션 생명주기 관리

    Startup: 모델 로드
    Shutdown: 리소스 정리
    """
    # Startup
    api_logger.info("[STARTUP] Food-101 API 서버 시작 중...")
    load_model()
    api_logger.info("[STARTUP] 서버 준비 완료!")

    yield

    # Shutdown
    api_logger.info("[SHUTDOWN] 서버 종료 중...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    api_logger.info("[SHUTDOWN] 리소스 정리 완료")


app = FastAPI(
    title="Food-101 Image Classification API",
    description="ResNet18 기반 음식 이미지 분류 API (리팩토링 버전)",
    version="2.0.0",
    lifespan=lifespan
)


# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Static 파일 마운트
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


# =============================================================================
# API 엔드포인트
# =============================================================================

@app.get("/")
async def root():
    """루트 엔드포인트 - 웹 인터페이스 또는 API 정보 제공"""
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        return FileResponse(str(html_path))
    else:
        return {
            "message": "Food-101 Image Classification API",
            "version": "2.0.0",
            "endpoints": {
                "predict": "POST /predict - 이미지 업로드 및 분류",
                "predict_gradcam": "POST /predict/gradcam - 분류 + Grad-CAM",
                "detect": "POST /detect - YOLO 객체 탐지",
                "health": "GET /health - 서버 상태 확인",
                "classes": "GET /classes - 분류 가능한 음식 목록",
                "docs": "GET /docs - API 문서 (Swagger UI)"
            }
        }


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """헬스 체크 엔드포인트"""
    is_healthy = MODEL is not None and DEVICE is not None

    return HealthCheckResponse(
        status="healthy" if is_healthy else "unhealthy",
        model_loaded=MODEL is not None,
        device=str(DEVICE) if DEVICE else None,
        num_classes=len(CLASSES)
    )


@app.get("/classes", response_model=ClassesResponse)
async def get_classes():
    """분류 가능한 음식 클래스 목록 반환"""
    return ClassesResponse(
        total=len(CLASSES),
        classes=CLASSES
    )


@app.post("/predict", response_model=ClassificationResponse)
async def predict(file: UploadFile = File(...)):
    """
    이미지를 업로드하여 음식 종류 예측 (ResNet18 기반)

    Args:
        file: 업로드된 이미지 파일 (JPEG, PNG 등)

    Returns:
        ClassificationResponse: 예측 결과 (Top-1, Top-5)
    """
    start_time = time()

    if MODEL is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다")

    try:
        # Step 1: 파일 검증 및 이미지 로드
        image = await validate_and_load_image(
            file,
            settings.max_file_size_bytes,
            settings.allowed_file_types_list
        )

        # Step 2: 이미지 전처리
        input_tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

        # Step 3: 예측 수행
        with torch.no_grad():
            outputs = MODEL(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            top5_prob, top5_idx = torch.topk(probabilities, 5)

            top1_idx = top5_idx[0].item()
            top1_prob = top5_prob[0].item()

        # Step 4: 응답 생성
        prediction = PredictionResult(
            class_name=CLASSES[top1_idx],
            class_id=top1_idx,
            confidence=float(top1_prob),
            confidence_percent=format_confidence(top1_prob)
        )

        top5 = [
            Top5Prediction(
                rank=i + 1,
                class_name=CLASSES[idx.item()],
                class_id=idx.item(),
                confidence=float(prob.item()),
                confidence_percent=format_confidence(prob.item())
            )
            for i, (prob, idx) in enumerate(zip(top5_prob, top5_idx))
        ]

        duration = time() - start_time
        log_prediction("ResNet18", top1_prob, duration)
        log_request("/predict", "POST", duration, 200)

        return ClassificationResponse(
            success=True,
            prediction=prediction,
            top5=top5
        )

    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"예측 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"예측 중 오류 발생: {str(e)}")


@app.post("/predict/gradcam", response_model=ClassificationResponse)
async def predict_with_gradcam(file: UploadFile = File(...)):
    """
    이미지를 업로드하여 음식 종류 예측 및 Grad-CAM 히트맵 생성

    Args:
        file: 업로드된 이미지 파일

    Returns:
        ClassificationResponse: 예측 결과 + Grad-CAM 히트맵
    """
    start_time = time()

    if MODEL is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다")

    try:
        # Step 1: 파일 검증 및 이미지 로드
        image = await validate_and_load_image(
            file,
            settings.max_file_size_bytes,
            settings.allowed_file_types_list
        )

        # Step 2: 이미지 전처리
        input_tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

        # Step 3: 예측 수행
        with torch.no_grad():
            outputs = MODEL(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            top5_prob, top5_idx = torch.topk(probabilities, 5)

            top1_idx = top5_idx[0].item()
            top1_prob = top5_prob[0].item()

        # Step 4: Grad-CAM 히트맵 생성
        overlay_image, pred_class, pred_prob = create_gradcam_visualization(
            MODEL, input_tensor, image, DEVICE
        )

        # Step 5: 히트맵 이미지를 base64로 인코딩
        buffered = io.BytesIO()
        overlay_image.save(buffered, format="PNG")
        heatmap_base64 = base64.b64encode(buffered.getvalue()).decode()

        # Step 6: 응답 생성
        prediction = PredictionResult(
            class_name=CLASSES[top1_idx],
            class_id=top1_idx,
            confidence=float(top1_prob),
            confidence_percent=format_confidence(top1_prob)
        )

        top5 = [
            Top5Prediction(
                rank=i + 1,
                class_name=CLASSES[idx.item()],
                class_id=idx.item(),
                confidence=float(prob.item()),
                confidence_percent=format_confidence(prob.item())
            )
            for i, (prob, idx) in enumerate(zip(top5_prob, top5_idx))
        ]

        gradcam = GradCAMResult(
            heatmap_image=f"data:image/png;base64,{heatmap_base64}",
            description="빨간색 영역이 모델이 판단할 때 중요하게 본 부분입니다"
        )

        duration = time() - start_time
        log_prediction("ResNet18+GradCAM", top1_prob, duration)
        log_request("/predict/gradcam", "POST", duration, 200)

        return ClassificationResponse(
            success=True,
            prediction=prediction,
            top5=top5,
            gradcam=gradcam
        )

    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Grad-CAM 예측 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"예측 중 오류 발생: {str(e)}")


@app.post("/detect", response_model=DetectionResponse)
async def detect_objects(
    file: UploadFile = File(...),
    conf_threshold: float = None
):
    """
    YOLO11n을 사용한 일반 객체 탐지 (80개 클래스)

    Args:
        file: 업로드된 이미지 파일
        conf_threshold: 신뢰도 임계값 (기본값: 설정 파일의 값)

    Returns:
        DetectionResponse: 탐지된 객체 정보 및 어노테이션된 이미지
    """
    start_time = time()

    if conf_threshold is None:
        conf_threshold = settings.yolo_confidence_threshold

    try:
        # Step 1: 파일 검증 및 이미지 로드
        image = await validate_and_load_image(
            file,
            settings.max_file_size_bytes,
            settings.allowed_file_types_list
        )

        # Step 2: YOLO 탐지기 인스턴스 가져오기
        detector = get_yolo_detector()

        # Step 3: 객체 탐지 수행
        result = detector.detect_and_encode(image, conf_threshold)

        # Step 4: 응답 생성
        detections = [
            Detection(
                class_name=det["class"],
                class_id=det["class_id"],
                confidence=det["confidence"],
                confidence_percent=det["confidence_percent"],
                bbox=BoundingBox(**det["bbox"])
            )
            for det in result['detections']
        ]

        duration = time() - start_time
        log_prediction("YOLO", 0.0, duration)
        log_request("/detect", "POST", duration, 200)

        return DetectionResponse(
            success=True,
            num_objects=result['num_objects'],
            detections=detections,
            annotated_image=result['annotated_image_base64']
        )

    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"YOLO 객체 탐지 실패: {e}", exc_info=True)
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
        api_logger.error(f"YOLO 클래스 목록 조회 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"클래스 목록 조회 중 오류 발생: {str(e)}")


# =============================================================================
# 서버 실행
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower()
    )
