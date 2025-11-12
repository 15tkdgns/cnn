"""
설정 관리 모듈
환경 변수와 기본값을 중앙화하여 관리
"""

from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    """
    애플리케이션 설정 클래스

    환경 변수에서 값을 자동으로 로드하며,
    없을 경우 기본값 사용
    """

    # 서버 설정
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    log_level: str = "INFO"

    # CORS 설정
    allowed_origins: str = "http://localhost:3000"

    # 모델 경로
    model_path: str = "./outputs/models/best_model.pth"
    classes_file_path: str = "./data/food-101/food-101/meta/classes.txt"
    dataset_path_file: str = "./data/dataset_path.txt"

    # 파일 업로드 제한
    max_file_size_mb: int = 10
    allowed_file_types: str = "image/jpeg,image/png,image/jpg,image/webp"

    # GPU 설정
    cuda_visible_devices: str = "0"
    use_mixed_precision: bool = True

    # YOLO 설정
    yolo_model: str = "yolo11n.pt"
    yolo_confidence_threshold: float = 0.25

    # 로깅 설정
    log_file: str = "./logs/api.log"
    log_rotation: str = "10MB"
    log_retention: str = "7 days"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    @property
    def max_file_size_bytes(self) -> int:
        """MB를 bytes로 변환"""
        return self.max_file_size_mb * 1024 * 1024

    @property
    def allowed_origins_list(self) -> List[str]:
        """쉼표로 구분된 origin을 리스트로 변환"""
        if self.allowed_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.allowed_origins.split(",")]

    @property
    def allowed_file_types_list(self) -> List[str]:
        """허용된 파일 타입 리스트"""
        return [ft.strip() for ft in self.allowed_file_types.split(",")]

    @property
    def model_path_resolved(self) -> Path:
        """절대 경로로 변환"""
        return Path(self.model_path).resolve()

    @property
    def classes_file_path_resolved(self) -> Path:
        """절대 경로로 변환"""
        return Path(self.classes_file_path).resolve()


@lru_cache()
def get_settings() -> Settings:
    """
    설정 싱글톤 인스턴스 반환

    @lru_cache 데코레이터로 한 번만 로드하여
    메모리 효율성 증가

    Returns:
        Settings: 설정 객체
    """
    return Settings()


# 편의를 위한 전역 인스턴스
settings = get_settings()


if __name__ == "__main__":
    # 설정 출력 (디버깅용)
    print("=" * 70)
    print("현재 설정:")
    print("=" * 70)
    print(f"Host: {settings.host}")
    print(f"Port: {settings.port}")
    print(f"Debug: {settings.debug}")
    print(f"Allowed Origins: {settings.allowed_origins_list}")
    print(f"Model Path: {settings.model_path_resolved}")
    print(f"Max File Size: {settings.max_file_size_mb}MB ({settings.max_file_size_bytes} bytes)")
    print(f"Allowed File Types: {settings.allowed_file_types_list}")
    print("=" * 70)
