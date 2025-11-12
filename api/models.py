"""
API 응답 모델 정의
Pydantic을 사용하여 타입 안전성 보장
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """바운딩 박스 좌표"""
    x1: float = Field(..., description="좌상단 X 좌표")
    y1: float = Field(..., description="좌상단 Y 좌표")
    x2: float = Field(..., description="우하단 X 좌표")
    y2: float = Field(..., description="우하단 Y 좌표")
    width: float = Field(..., description="박스 너비")
    height: float = Field(..., description="박스 높이")


class PredictionResult(BaseModel):
    """단일 예측 결과"""
    class_name: str = Field(..., alias="class", description="예측된 클래스 이름")
    class_id: int = Field(..., description="클래스 ID (0~100)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="신뢰도 (0~1)")
    confidence_percent: str = Field(..., description="신뢰도 퍼센트 (예: '85.23%')")

    class Config:
        populate_by_name = True


class Top5Prediction(BaseModel):
    """Top-5 예측 결과"""
    rank: int = Field(..., ge=1, le=5, description="순위 (1~5)")
    class_name: str = Field(..., alias="class", description="클래스 이름")
    class_id: int = Field(..., description="클래스 ID")
    confidence: float = Field(..., ge=0.0, le=1.0, description="신뢰도")
    confidence_percent: str = Field(..., description="신뢰도 퍼센트")

    class Config:
        populate_by_name = True


class GradCAMResult(BaseModel):
    """Grad-CAM 히트맵 결과"""
    heatmap_image: str = Field(..., description="base64 인코딩된 히트맵 이미지 (Data URI)")
    description: str = Field(..., description="히트맵 설명")


class ClassificationResponse(BaseModel):
    """음식 분류 API 응답"""
    success: bool = Field(True, description="성공 여부")
    prediction: PredictionResult = Field(..., description="메인 예측 결과")
    top5: List[Top5Prediction] = Field(..., description="상위 5개 예측 결과")
    gradcam: Optional[GradCAMResult] = Field(None, description="Grad-CAM 히트맵 (선택사항)")


class Detection(BaseModel):
    """객체 탐지 결과 (단일 객체)"""
    class_name: str = Field(..., alias="class", description="객체 클래스 이름")
    class_id: int = Field(..., description="객체 클래스 ID")
    confidence: float = Field(..., ge=0.0, le=1.0, description="신뢰도")
    confidence_percent: str = Field(..., description="신뢰도 퍼센트")
    bbox: BoundingBox = Field(..., description="바운딩 박스 좌표")

    class Config:
        populate_by_name = True


class DetectionResponse(BaseModel):
    """객체 탐지 API 응답"""
    success: bool = Field(True, description="성공 여부")
    num_objects: int = Field(..., ge=0, description="탐지된 객체 수")
    detections: List[Detection] = Field(..., description="탐지된 객체 리스트")
    annotated_image: str = Field(..., description="어노테이션된 이미지 (base64 Data URI)")


class HealthCheckResponse(BaseModel):
    """헬스 체크 응답"""
    status: str = Field(..., description="상태 ('healthy' 또는 'unhealthy')")
    model_loaded: bool = Field(..., description="모델 로드 여부")
    device: Optional[str] = Field(None, description="연산 디바이스 (cuda/cpu)")
    num_classes: int = Field(..., description="클래스 수")


class ClassesResponse(BaseModel):
    """클래스 목록 응답"""
    total: int = Field(..., description="전체 클래스 수")
    classes: List[str] = Field(..., description="클래스 이름 리스트")


class ErrorResponse(BaseModel):
    """에러 응답"""
    detail: str = Field(..., description="에러 메시지")
    error_code: Optional[str] = Field(None, description="에러 코드")
