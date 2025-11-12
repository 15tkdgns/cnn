"""
유틸리티 함수 모듈
재사용 가능한 헬퍼 함수들
"""

from pathlib import Path
from typing import List
import logging
from PIL import Image
from fastapi import UploadFile, HTTPException

logger = logging.getLogger(__name__)


def load_classes(classes_file_path: Path, dataset_path_file: Path) -> List[str]:
    """
    Food-101 데이터셋의 클래스 이름 목록을 로드

    Args:
        classes_file_path: classes.txt 파일 경로
        dataset_path_file: dataset_path.txt 파일 경로

    Returns:
        list: 101개 음식 클래스 이름

    Examples:
        >>> classes = load_classes(Path("classes.txt"), Path("dataset_path.txt"))
        >>> print(classes[0])
        'apple_pie'
    """
    try:
        # dataset_path.txt가 있으면 그 경로를 우선 사용
        if dataset_path_file.exists():
            with open(dataset_path_file, 'r') as f:
                base_path = Path(f.read().strip())
            classes_file = base_path / "food-101" / "food-101" / "meta" / "classes.txt"
        else:
            classes_file = classes_file_path

        # classes.txt 파일이 없으면 기본 클래스 이름 사용
        if not classes_file.exists():
            logger.warning(f"클래스 파일을 찾을 수 없습니다: {classes_file}")
            logger.warning("기본 클래스 목록을 사용합니다 (class_0 ~ class_100)")
            return [f"class_{i}" for i in range(101)]

        # 클래스 파일 읽기
        with open(classes_file, 'r') as f:
            classes = [line.strip() for line in f]

        logger.info(f"[SUCCESS] {len(classes)}개 클래스 로드 완료")
        return classes

    except Exception as e:
        logger.error(f"클래스 로드 실패: {e}")
        # 예외 발생 시 기본 클래스 목록 반환
        return [f"class_{i}" for i in range(101)]


def validate_file_size(file_size: int, max_size: int) -> None:
    """
    파일 크기 검증

    Args:
        file_size: 파일 크기 (bytes)
        max_size: 최대 허용 크기 (bytes)

    Raises:
        HTTPException: 파일이 너무 큰 경우
    """
    if file_size > max_size:
        max_mb = max_size / (1024 * 1024)
        current_mb = file_size / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"파일 크기가 제한을 초과했습니다. (최대: {max_mb:.1f}MB, 현재: {current_mb:.1f}MB)"
        )


def validate_file_type(content_type: str, allowed_types: List[str]) -> None:
    """
    파일 타입 검증

    Args:
        content_type: MIME 타입 (예: 'image/jpeg')
        allowed_types: 허용된 MIME 타입 리스트

    Raises:
        HTTPException: 허용되지 않은 파일 타입인 경우
    """
    if content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail=f"지원하지 않는 파일 형식입니다. 허용된 타입: {', '.join(allowed_types)}"
        )


async def validate_and_load_image(
    file: UploadFile,
    max_size: int,
    allowed_types: List[str]
) -> Image.Image:
    """
    파일 검증 및 PIL Image로 로드

    Args:
        file: 업로드된 파일
        max_size: 최대 파일 크기 (bytes)
        allowed_types: 허용된 MIME 타입 리스트

    Returns:
        PIL.Image: RGB 모드 이미지

    Raises:
        HTTPException: 검증 실패 시
    """
    # MIME 타입 검증
    validate_file_type(file.content_type, allowed_types)

    # 파일 읽기
    image_bytes = await file.read()

    # 파일 크기 검증
    validate_file_size(len(image_bytes), max_size)

    # PIL Image로 변환
    try:
        from io import BytesIO
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        return image
    except Exception as e:
        logger.error(f"이미지 로드 실패: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"이미지 파일을 읽을 수 없습니다: {str(e)}"
        )


def format_confidence(confidence: float) -> str:
    """
    신뢰도를 퍼센트 문자열로 포맷

    Args:
        confidence: 신뢰도 (0~1)

    Returns:
        str: 퍼센트 문자열 (예: "85.23%")

    Examples:
        >>> format_confidence(0.8523)
        '85.23%'
    """
    return f"{confidence * 100:.2f}%"


def ensure_directory_exists(directory: Path) -> None:
    """
    디렉토리가 존재하지 않으면 생성

    Args:
        directory: 생성할 디렉토리 경로
    """
    directory.mkdir(parents=True, exist_ok=True)
