"""
로깅 설정 모듈
구조화된 로깅 및 파일 회전 지원
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from datetime import datetime
from .config import settings


def setup_logger(name: str = "api") -> logging.Logger:
    """
    로거 설정 및 반환

    Args:
        name: 로거 이름

    Returns:
        logging.Logger: 설정된 로거
    """
    # 로거 생성
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, settings.log_level.upper()))

    # 기존 핸들러 제거 (중복 방지)
    logger.handlers.clear()

    # 포맷터 설정
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 파일 핸들러 (회전 로깅)
    if settings.log_file:
        try:
            log_path = Path(settings.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            # 로그 회전 설정
            max_bytes = _parse_size(settings.log_rotation)
            backup_count = 5

            file_handler = RotatingFileHandler(
                filename=log_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        except Exception as e:
            logger.warning(f"파일 로깅 설정 실패: {e}")

    return logger


def _parse_size(size_str: str) -> int:
    """
    크기 문자열을 bytes로 변환

    Args:
        size_str: 크기 문자열 (예: '10MB', '1GB')

    Returns:
        int: bytes 단위 크기
    """
    size_str = size_str.upper().strip()

    if size_str.endswith('GB'):
        return int(float(size_str[:-2]) * 1024 * 1024 * 1024)
    elif size_str.endswith('MB'):
        return int(float(size_str[:-2]) * 1024 * 1024)
    elif size_str.endswith('KB'):
        return int(float(size_str[:-2]) * 1024)
    else:
        return int(size_str)


# 전역 로거 인스턴스
api_logger = setup_logger("api")


# 요청 로깅 헬퍼 함수
def log_request(endpoint: str, method: str, duration: float, status_code: int):
    """
    API 요청 로깅

    Args:
        endpoint: 엔드포인트 경로
        method: HTTP 메서드
        duration: 처리 시간 (초)
        status_code: HTTP 상태 코드
    """
    level = logging.INFO if status_code < 400 else logging.ERROR
    api_logger.log(
        level,
        f"{method} {endpoint} - {status_code} - {duration:.3f}s"
    )


def log_prediction(model_type: str, confidence: float, duration: float):
    """
    예측 로깅

    Args:
        model_type: 모델 타입 ('ResNet18' 또는 'YOLO')
        confidence: 신뢰도
        duration: 추론 시간 (초)
    """
    api_logger.info(
        f"{model_type} prediction - confidence: {confidence:.4f} - {duration:.3f}s"
    )
