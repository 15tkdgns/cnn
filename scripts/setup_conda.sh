#!/bin/bash
# ============================================================================
# Food-101 프로젝트 conda 가상환경 설정 스크립트
# ============================================================================
#
# 사용법:
#   chmod +x setup_conda.sh
#   ./setup_conda.sh
#
# 이 스크립트는 다음 작업을 수행합니다:
# 1. conda 환경 생성 (food101)
# 2. PyTorch + CUDA 설치
# 3. 필요한 패키지 설치
# 4. 설치 확인
# ============================================================================

set -e  # 오류 발생 시 스크립트 중단

echo "======================================================================"
echo "Food-101 프로젝트 conda 환경 설정 시작"
echo "======================================================================"

# ----------------------------------------------------------------------------
# 0단계: conda 설치 확인
# ----------------------------------------------------------------------------
echo ""
echo "0. conda 설치 확인 중..."

if ! command -v conda &> /dev/null; then
    echo "   ❌ conda가 설치되어 있지 않습니다."
    echo ""
    echo "   다음 명령어로 Miniconda를 먼저 설치하세요:"
    echo "     chmod +x install_miniconda.sh"
    echo "     ./install_miniconda.sh"
    echo "     source ~/.bashrc"
    exit 1
fi

# conda 초기화 (현재 세션에서)
eval "$(conda shell.bash hook)"

conda_version=$(conda --version)
echo "   ✓ conda 확인: $conda_version"

# ----------------------------------------------------------------------------
# 1단계: 기존 환경 확인
# ----------------------------------------------------------------------------
echo ""
echo "1. 기존 conda 환경 확인 중..."

if conda env list | grep -q "^food101 "; then
    echo "   ⚠️  'food101' 환경이 이미 존재합니다."
    read -p "   삭제하고 새로 생성하시겠습니까? (y/N): " confirm

    if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
        echo "   기존 환경 삭제 중..."
        conda env remove -n food101 -y
        echo "   ✓ 삭제 완료"
    else
        echo "   기존 환경을 사용합니다."
        conda activate food101
        echo "   ✓ 환경 활성화 완료"
        exit 0
    fi
fi

# ----------------------------------------------------------------------------
# 2단계: conda 환경 생성
# ----------------------------------------------------------------------------
echo ""
echo "2. conda 환경 생성 중 (Python 3.10)..."
echo "   환경 이름: food101"

conda create -n food101 python=3.10 -y

echo "   ✓ 환경 생성 완료"

# ----------------------------------------------------------------------------
# 3단계: 환경 활성화
# ----------------------------------------------------------------------------
echo ""
echo "3. 환경 활성화 중..."

conda activate food101

echo "   ✓ 환경 활성화 완료"

# ----------------------------------------------------------------------------
# 4단계: GPU 확인 및 PyTorch 설치
# ----------------------------------------------------------------------------
echo ""
echo "4. PyTorch 설치 중..."

# NVIDIA GPU 확인
if command -v nvidia-smi &> /dev/null; then
    echo "   ✓ NVIDIA GPU 감지됨"
    nvidia-smi --query-gpu=name --format=csv,noheader | head -1

    echo ""
    echo "   PyTorch (GPU 버전) 설치 중... (시간이 걸릴 수 있습니다)"
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

    echo "   ✓ PyTorch (GPU) 설치 완료"
else
    echo "   ⚠️  NVIDIA GPU를 찾을 수 없습니다. CPU 버전을 설치합니다."

    echo ""
    echo "   PyTorch (CPU 버전) 설치 중..."
    conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

    echo "   ✓ PyTorch (CPU) 설치 완료"
fi

# ----------------------------------------------------------------------------
# 5단계: 추가 패키지 설치
# ----------------------------------------------------------------------------
echo ""
echo "5. 추가 패키지 설치 중..."

# pip 업그레이드
pip install --upgrade pip setuptools wheel --quiet

# 필요한 패키지 설치
echo "   - kagglehub 설치 중..."
pip install kagglehub --quiet

echo "   - 데이터 분석 라이브러리 설치 중..."
pip install pandas numpy --quiet

echo "   - 시각화 라이브러리 설치 중..."
pip install matplotlib seaborn --quiet

echo "   - 유틸리티 설치 중..."
pip install tqdm pillow --quiet

echo "   - Jupyter 설치 중..."
pip install jupyter ipykernel ipywidgets --quiet

echo "   ✓ 모든 패키지 설치 완료"

# ----------------------------------------------------------------------------
# 6단계: Jupyter 커널 등록
# ----------------------------------------------------------------------------
echo ""
echo "6. Jupyter 커널 등록 중..."

python -m ipykernel install --user --name food101 --display-name "Python (food101)"

echo "   ✓ Jupyter 커널 등록 완료"

# ----------------------------------------------------------------------------
# 7단계: 설치 확인
# ----------------------------------------------------------------------------
echo ""
echo "7. 설치 확인 중..."

echo ""
echo "--- Python 버전 ---"
python --version

echo ""
echo "--- PyTorch 정보 ---"
python -c "
import torch
print(f'PyTorch 버전: {torch.__version__}')
print(f'CUDA 사용 가능: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA 버전: {torch.version.cuda}')
    print(f'GPU 장치 수: {torch.cuda.device_count()}')
    print(f'현재 GPU: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "--- 설치된 주요 패키지 ---"
python -c "
import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
import PIL
print(f'NumPy: {np.__version__}')
print(f'Pandas: {pd.__version__}')
print(f'Matplotlib: {matplotlib.__version__}')
print(f'Seaborn: {sns.__version__}')
print(f'Pillow: {PIL.__version__}')
"

# ----------------------------------------------------------------------------
# 완료 메시지
# ----------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo "✓ Food-101 conda 환경 설정이 완료되었습니다!"
echo "======================================================================"
echo ""
echo "📌 환경 사용 방법:"
echo ""
echo "1. 환경 활성화:"
echo "   conda activate food101"
echo ""
echo "2. 데이터셋 다운로드:"
echo "   python download_dataset.py"
echo ""
echo "3. EDA 노트북 실행:"
echo "   jupyter notebook food101_eda.ipynb"
echo "   (노트북에서 커널을 'Python (food101)'로 선택하세요)"
echo ""
echo "4. 환경 비활성화:"
echo "   conda deactivate"
echo ""
echo "======================================================================"
