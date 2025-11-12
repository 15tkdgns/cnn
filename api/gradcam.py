"""
Grad-CAM (Gradient-weighted Class Activation Mapping) 구현
이미지 분류 모델의 판단 근거를 시각화하는 히트맵 생성

Grad-CAM이란?
- CNN 모델이 이미지의 어느 부분을 보고 판단했는지 시각화
- 클래스별 그래디언트를 활용하여 중요한 영역을 강조
- 빨간색/노란색: 해당 클래스 판단에 중요한 영역 (양의 기여도)
- 파란색: 중요도가 낮은 영역

참고 논문:
Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization", ICCV 2017
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2


class GradCAM:
    """
    Grad-CAM을 사용하여 CNN 모델의 판단 근거를 시각화

    동작 원리:
    1. Forward pass로 타겟 레이어의 activation을 캡처
    2. Backward pass로 타겟 클래스에 대한 gradient를 계산
    3. Gradient의 global average pooling으로 채널별 가중치 계산
    4. 가중치와 activation을 결합하여 히트맵 생성

    Args:
        model: PyTorch 모델
        target_layer: 시각화할 대상 레이어 (일반적으로 마지막 Conv 레이어)
                     ResNet18의 경우 layer4[-1]이 적합
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None     # Backward pass에서 캡처한 그래디언트
        self.activations = None   # Forward pass에서 캡처한 활성화 맵

        # PyTorch Hook 등록 (중간층 데이터 캡처용)
        self._register_hooks()

    def _register_hooks(self):
        """
        Forward/Backward Hook 등록

        PyTorch Hook이란?
        - 모델의 중간층 데이터(activation, gradient)를 캡처하는 메커니즘
        - 일반적으로 접근 불가능한 중간층 정보를 가로채서 저장
        - Grad-CAM은 마지막 Conv 레이어의 activation과 gradient가 필요

        등록되는 Hook:
        1. Forward Hook: 순전파 시 activation 맵 저장
        2. Backward Hook: 역전파 시 gradient 저장
        """

        def forward_hook(module, input, output):
            """
            Forward pass에서 activation 저장

            Args:
                module: 타겟 레이어 (self.target_layer)
                input: 레이어 입력 (사용 안 함)
                output: 레이어 출력 (activation 맵)
                       Shape: (batch_size, channels, height, width)
            """
            # detach()로 그래프에서 분리하여 메모리 절약
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            """
            Backward pass에서 gradient 저장

            Args:
                module: 타겟 레이어
                grad_input: 레이어 입력에 대한 gradient (사용 안 함)
                grad_output: 레이어 출력에 대한 gradient
                            Shape: (batch_size, channels, height, width)
            """
            # grad_output은 튜플이므로 [0]으로 첫 번째 요소 추출
            self.gradients = grad_output[0].detach()

        # Hook 등록: 레이어 연산 시 자동으로 위 함수들이 호출됨
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, target_class=None):
        """
        Grad-CAM 히트맵 생성

        수학적 공식:
        L^c_GradCAM = ReLU(Σ_k α^c_k * A^k)

        여기서:
        - α^c_k = (1/Z) * Σ_i Σ_j ∂y^c/∂A^k_ij  (채널 k의 가중치)
        - A^k: 채널 k의 activation 맵
        - y^c: 클래스 c의 점수 (logit)
        - Z: 공간 차원의 크기 (H * W)

        Args:
            input_tensor: 입력 이미지 텐서 (1, C, H, W)
            target_class: 시각화할 클래스 인덱스 (None이면 예측 클래스 사용)

        Returns:
            tuple: (
                numpy.ndarray: Grad-CAM 히트맵 (H, W) 범위 [0, 1],
                int: 사용된 타겟 클래스 인덱스
            )

        동작 과정:
        1. Forward pass로 모델 예측 수행 (activation 자동 캡처)
        2. 타겟 클래스의 점수에 대해 backward pass (gradient 자동 캡처)
        3. Gradient를 공간 차원(H, W)에 대해 평균내어 채널별 가중치 계산
        4. 가중치와 activation을 곱하고 채널 방향으로 합산
        5. ReLU로 양수만 남김 (음수는 클래스 판단에 방해되는 영역)
        6. 0~1 범위로 정규화
        """
        # 모델을 평가 모드로 설정 (Dropout, BatchNorm 비활성화)
        self.model.eval()

        # Step 1: Forward pass
        # Hook이 자동으로 self.activations에 타겟 레이어 출력 저장
        output = self.model(input_tensor)  # Shape: (1, num_classes)

        # Step 2: 타겟 클래스 결정
        # None이면 모델이 예측한 클래스 사용
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Step 3: Backward pass (타겟 클래스의 점수에 대해)
        # Hook이 자동으로 self.gradients에 타겟 레이어 gradient 저장
        self.model.zero_grad()                    # 기존 gradient 초기화
        class_score = output[0, target_class]     # 타겟 클래스의 점수 (scalar)
        class_score.backward()                    # 역전파 수행

        # Step 4: Grad-CAM 계산
        # 4-1. Gradient의 Global Average Pooling (채널별 중요도 가중치)
        # gradients shape: (1, C, H, W) -> weights shape: (1, C, 1, 1)
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)

        # 4-2. 가중치와 activation의 가중합 (채널 방향으로)
        # (1, C, 1, 1) * (1, C, H, W) -> (1, C, H, W)
        # 그 후 sum(dim=1) -> (1, 1, H, W)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)

        # 4-3. ReLU 적용 (양수 영향만 남김)
        # 음수 영역은 해당 클래스 판단을 방해하는 부분이므로 제거
        cam = F.relu(cam)

        # 4-4. 정규화 [0, 1] 범위로 스케일링
        cam = cam.squeeze().cpu().numpy()  # (H, W) numpy array로 변환
        cam_min = cam.min()
        cam_max = cam.max()
        # 1e-8은 division by zero 방지
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam, target_class

    def generate_heatmap_overlay(self, input_image, cam, alpha=0.4):
        """
        원본 이미지에 히트맵 오버레이

        Args:
            input_image: PIL Image 또는 numpy array (H, W, 3) RGB 형식
            cam: Grad-CAM 히트맵 (H', W') 범위 [0, 1]
            alpha: 히트맵 투명도 (0~1)
                  0에 가까울수록 원본 이미지가 선명
                  1에 가까울수록 히트맵이 선명
                  기본값 0.4는 적절한 균형

        Returns:
            PIL.Image: 히트맵이 오버레이된 이미지 (H, W, 3) RGB

        동작 과정:
        1. PIL Image를 numpy array로 변환
        2. CAM을 원본 이미지 크기로 리사이즈
        3. JET 컬러맵 적용 (파란색→초록색→노란색→빨간색)
        4. 히트맵과 원본 이미지를 alpha 비율로 블렌딩
        """
        # Step 1: PIL Image를 numpy array로 변환
        if isinstance(input_image, Image.Image):
            img = np.array(input_image)  # (H, W, 3) uint8 RGB
        else:
            img = input_image

        # Step 2: CAM을 원본 이미지 크기로 리사이즈
        # cam: (7, 7) -> (H, W) 예: (224, 224)
        h, w = img.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))

        # Step 3: 히트맵 생성 (JET 컬러맵)
        # [0, 1] -> [0, 255] 스케일링
        # JET 컬러맵: 낮은 값(파란색) -> 높은 값(빨간색)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        # OpenCV는 BGR이므로 RGB로 변환
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Step 4: 히트맵과 원본 이미지 블렌딩
        # overlay = α * heatmap + (1-α) * image
        overlay = heatmap * alpha + img * (1 - alpha)
        overlay = np.uint8(overlay)  # float -> uint8 변환

        # PIL Image로 변환하여 반환
        return Image.fromarray(overlay)


def get_gradcam_for_resnet18(model):
    """
    ResNet18용 Grad-CAM 객체 생성

    ResNet18 구조:
    - layer1: Conv block 1 (64 channels)
    - layer2: Conv block 2 (128 channels)
    - layer3: Conv block 3 (256 channels)
    - layer4: Conv block 4 (512 channels) <- 마지막 Conv 레이어
    - avgpool: Global Average Pooling
    - fc: Fully Connected Layer

    Grad-CAM은 마지막 Conv 레이어(layer4[-1])를 사용해야
    공간 정보를 유지하면서 고수준 특징을 시각화할 수 있음

    Args:
        model: ResNet18 모델

    Returns:
        GradCAM: Grad-CAM 객체 (layer4[-1]을 타겟으로 설정)
    """
    # ResNet18의 마지막 컨볼루션 레이어
    # layer4[-1]은 BasicBlock의 마지막 블록
    target_layer = model.layer4[-1]
    return GradCAM(model, target_layer)


def create_gradcam_visualization(model, input_tensor, original_image, device):
    """
    Grad-CAM 시각화 이미지 생성 (편의 함수)

    API 엔드포인트에서 한 번의 호출로 Grad-CAM 시각화를 완성하기 위한 래퍼 함수

    Args:
        model: PyTorch 모델 (ResNet18)
        input_tensor: 전처리된 입력 텐서 (1, 3, 224, 224)
        original_image: 원본 PIL 이미지 (히트맵 오버레이용)
        device: 연산 디바이스 (cuda 또는 cpu)

    Returns:
        tuple: (
            PIL.Image: 히트맵이 오버레이된 이미지,
            int: 예측 클래스 인덱스,
            float: 예측 확률 (0~1)
        )

    동작 과정:
    1. ResNet18용 Grad-CAM 객체 생성
    2. 입력 텐서를 디바이스로 이동
    3. Grad-CAM 히트맵 생성
    4. 예측 확률 계산
    5. 원본 이미지에 히트맵 오버레이
    """
    # Step 1: Grad-CAM 객체 생성 (layer4[-1]을 타겟으로)
    gradcam = get_gradcam_for_resnet18(model)

    # Step 2: 입력을 디바이스로 이동
    input_tensor = input_tensor.to(device)

    # Step 3: CAM 생성 (히트맵 및 예측 클래스)
    cam, pred_class = gradcam.generate_cam(input_tensor)

    # Step 4: 예측 확률 계산
    with torch.no_grad():  # 추론 모드 (gradient 불필요)
        output = model(input_tensor)  # (1, 101) logits
        probs = torch.softmax(output, dim=1)  # (1, 101) probabilities
        pred_prob = probs[0, pred_class].item()  # float 값으로 변환

    # Step 5: 히트맵 오버레이 생성
    # alpha=0.4로 히트맵과 원본 이미지를 적절히 블렌딩
    overlay_image = gradcam.generate_heatmap_overlay(original_image, cam, alpha=0.4)

    return overlay_image, pred_class, pred_prob
