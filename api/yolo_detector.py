"""
YOLO Object Detection Wrapper
사전학습된 YOLO 모델을 사용한 객체 탐지

YOLO (You Only Look Once):
- 실시간 객체 탐지를 위한 딥러닝 모델
- 한 번의 forward pass로 모든 객체를 탐지 (빠른 속도)
- COCO 데이터셋으로 사전학습 (80개 클래스)

COCO 데이터셋 주요 클래스:
- 사람, 동물 (person, cat, dog, horse, ...)
- 차량 (car, bicycle, bus, truck, ...)
- 가구 (chair, table, bed, ...)
- 전자기기 (laptop, tv, keyboard, ...)
"""

from ultralytics import YOLO
from PIL import Image
import numpy as np
import io
import base64


class YOLODetector:
    """
    YOLO 객체 탐지 래퍼 클래스

    주요 기능:
    - 80개의 COCO 클래스 객체 탐지
    - 바운딩 박스 좌표 반환 (x1, y1, x2, y2)
    - 클래스 이름 및 신뢰도 반환
    - 어노테이션된 이미지 생성 (바운딩 박스 + 라벨)

    지원 모델:
    - yolo11n.pt: YOLO v11 Nano (빠르고 가벼움)
    - yolov8n.pt: YOLO v8 Nano
    - yolov8s.pt: YOLO v8 Small (더 정확하지만 느림)
    """

    def __init__(self, model_name='yolo11n.pt'):
        """
        YOLO 모델 초기화

        Args:
            model_name: YOLO 모델 파일 이름 (기본: yolo11n.pt)
                       처음 실행 시 자동으로 다운로드됨

        동작:
        1. Ultralytics YOLO 라이브러리로 모델 로드
        2. 모델의 클래스 이름 딕셔너리 저장 (0: 'person', 1: 'bicycle', ...)
        """
        self.model = YOLO(model_name)  # 모델 로드 (없으면 자동 다운로드)
        self.class_names = self.model.names  # {0: 'person', 1: 'bicycle', ...}

    def detect(self, image, conf_threshold=0.25):
        """
        이미지에서 객체 탐지

        Args:
            image: PIL Image 또는 numpy array (H, W, 3) RGB
            conf_threshold: 신뢰도 임계값 (0.0 ~ 1.0, 기본: 0.25)
                           이 값보다 높은 신뢰도의 객체만 반환

        Returns:
            dict: {
                'detections': [
                    {
                        'class': 'person',           # 클래스 이름
                        'class_id': 0,               # 클래스 ID
                        'confidence': 0.89,          # 신뢰도
                        'confidence_percent': '89.00%',
                        'bbox': {
                            'x1': 100, 'y1': 50,     # 좌상단 (x, y)
                            'x2': 300, 'y2': 400,    # 우하단 (x, y)
                            'width': 200,            # 박스 너비
                            'height': 350            # 박스 높이
                        }
                    },
                    ...
                ],
                'num_objects': 3,                    # 탐지된 객체 수
                'annotated_image': PIL.Image        # 바운딩 박스가 그려진 이미지
            }

        동작 과정:
        1. PIL Image를 numpy array로 변환
        2. YOLO 모델로 객체 탐지 수행
        3. 바운딩 박스, 클래스, 신뢰도 정보 추출
        4. 어노테이션된 이미지 생성 (바운딩 박스 + 라벨)
        5. BGR → RGB 변환 (OpenCV → PIL)
        """
        # Step 1: PIL Image를 numpy array로 변환
        # YOLO는 numpy array, PIL Image, 파일 경로 모두 지원
        # 하지만 일관성을 위해 numpy array로 변환
        if isinstance(image, Image.Image):
            image = np.array(image)  # (H, W, 3) uint8 RGB

        # Step 2: YOLO 예측 수행
        results = self.model.predict(
            source=image,                # 입력 이미지
            conf=conf_threshold,         # 신뢰도 임계값
            verbose=False                # 출력 로그 비활성화
        )

        result = results[0]  # 배치 크기가 1이므로 첫 번째 결과만 사용

        # Step 3: 탐지된 객체 정보 추출
        detections = []

        if len(result.boxes) > 0:  # 탐지된 객체가 있는 경우
            # 바운딩 박스 좌표 (xyxy 형식: x1, y1, x2, y2)
            boxes = result.boxes.xyxy.cpu().numpy()  # (N, 4)
            # 신뢰도 점수
            confidences = result.boxes.conf.cpu().numpy()  # (N,)
            # 클래스 ID
            class_ids = result.boxes.cls.cpu().numpy().astype(int)  # (N,)

            # 각 탐지 결과를 딕셔너리로 변환
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = box
                detections.append({
                    'class': self.class_names[cls_id],  # 클래스 이름 매핑
                    'class_id': int(cls_id),
                    'confidence': float(conf),
                    'confidence_percent': f"{conf * 100:.2f}%",
                    'bbox': {
                        'x1': float(x1),  # 좌상단 x
                        'y1': float(y1),  # 좌상단 y
                        'x2': float(x2),  # 우하단 x
                        'y2': float(y2),  # 우하단 y
                        'width': float(x2 - x1),   # 박스 너비
                        'height': float(y2 - y1)   # 박스 높이
                    }
                })

        # Step 4: 어노테이션된 이미지 생성
        # plot() 메서드가 바운딩 박스와 라벨을 자동으로 그려줌
        annotated_img = result.plot()  # BGR numpy array (OpenCV 형식)

        # Step 5: BGR to RGB 변환
        # OpenCV는 BGR, PIL은 RGB 사용
        # [..., ::-1]은 마지막 차원(채널)을 역순으로 (BGR -> RGB)
        annotated_img_rgb = annotated_img[..., ::-1]
        annotated_pil = Image.fromarray(annotated_img_rgb)

        return {
            'detections': detections,
            'num_objects': len(detections),
            'annotated_image': annotated_pil
        }

    def detect_and_encode(self, image, conf_threshold=0.25):
        """
        객체 탐지 및 결과 이미지를 base64로 인코딩

        API 응답에서 이미지를 JSON으로 전송하기 위한 편의 함수

        Args:
            image: PIL Image
            conf_threshold: 신뢰도 임계값 (0.0 ~ 1.0)

        Returns:
            dict: {
                'detections': [...],                     # 탐지 결과 리스트
                'num_objects': 3,                        # 탐지된 객체 수
                'annotated_image_base64': 'data:image/png;base64,...'
            }

        동작 과정:
        1. detect() 메서드로 객체 탐지 수행
        2. 어노테이션된 이미지를 PNG로 저장 (메모리 버퍼)
        3. base64로 인코딩
        4. Data URI 형식으로 반환 (프론트엔드에서 <img src=...> 사용 가능)
        """
        # Step 1: 객체 탐지 수행
        result = self.detect(image, conf_threshold)

        # Step 2: 어노테이션된 이미지를 base64로 인코딩
        # 메모리 버퍼에 PNG로 저장
        buffered = io.BytesIO()
        result['annotated_image'].save(buffered, format="PNG")

        # base64로 인코딩 후 문자열로 변환
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        # Step 3: Data URI 형식으로 반환
        # 프론트엔드에서 <img src="data:image/png;base64,..."> 형태로 사용
        return {
            'detections': result['detections'],
            'num_objects': result['num_objects'],
            'annotated_image_base64': f"data:image/png;base64,{img_base64}"
        }

    def get_class_names(self):
        """
        모델이 탐지할 수 있는 클래스 이름 반환

        Returns:
            dict: {0: 'person', 1: 'bicycle', 2: 'car', ..., 79: 'toothbrush'}
                 총 80개의 COCO 클래스
        """
        return self.class_names


# 전역 YOLO 인스턴스 (지연 로딩)
# 서버 시작 시가 아니라 첫 요청 시 로드 (startup 시간 단축)
_yolo_instance = None


def get_yolo_detector():
    """
    YOLO 탐지기 인스턴스 반환 (싱글톤 패턴)

    싱글톤 패턴을 사용하여 메모리 효율성 향상:
    - 여러 요청에서 동일한 YOLO 인스턴스 공유
    - 모델을 한 번만 로드하여 메모리 절약
    - 첫 요청 시에만 로딩 시간이 발생

    Returns:
        YOLODetector: 전역 YOLO 탐지기 인스턴스

    사용 예:
        detector = get_yolo_detector()
        result = detector.detect(image)
    """
    global _yolo_instance
    if _yolo_instance is None:
        # 첫 호출 시에만 YOLO 모델 로드
        _yolo_instance = YOLODetector()
    return _yolo_instance
