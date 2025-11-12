# ============================================================================
# venv 가상환경 자동 설정 스크립트 (Windows PowerShell)
# ============================================================================
#
# 사용법:
#   PowerShell에서 실행:
#   .\setup_venv.ps1
#
# 만약 실행 정책 오류가 발생하면 다음 명령어를 먼저 실행하세요:
#   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
#
# 이 스크립트는 다음 작업을 수행합니다:
# 1. venv 가상환경 생성
# 2. 가상환경 활성화
# 3. pip 업그레이드
# 4. requirements.txt의 패키지 설치
# ============================================================================

# 오류 발생 시 스크립트 중단
$ErrorActionPreference = "Stop"

Write-Host "======================================================================"
Write-Host "Food-101 프로젝트 가상환경 설정 시작 (venv)"
Write-Host "======================================================================"

# ----------------------------------------------------------------------------
# 1단계: Python 버전 확인
# ----------------------------------------------------------------------------
Write-Host ""
Write-Host "1. Python 버전 확인 중..."

try {
    $pythonVersion = python --version
    Write-Host "   $pythonVersion" -ForegroundColor Green

    # Python 3.8 이상 확인
    $versionMatch = $pythonVersion -match 'Python (\d+)\.(\d+)'
    if ($versionMatch) {
        $major = [int]$matches[1]
        $minor = [int]$matches[2]

        if (($major -lt 3) -or (($major -eq 3) -and ($minor -lt 8))) {
            Write-Host "   ⚠️  Python 3.8 이상이 필요합니다." -ForegroundColor Red
            exit 1
        }
    }

    Write-Host "   ✓ Python 버전 확인 완료" -ForegroundColor Green
}
catch {
    Write-Host "   ❌ Python이 설치되어 있지 않습니다." -ForegroundColor Red
    Write-Host "   https://www.python.org/downloads/ 에서 Python을 설치하세요."
    exit 1
}

# ----------------------------------------------------------------------------
# 2단계: 기존 가상환경 확인
# ----------------------------------------------------------------------------
Write-Host ""
Write-Host "2. 기존 가상환경 확인 중..."

if (Test-Path "venv") {
    Write-Host "   ⚠️  기존 venv 폴더가 발견되었습니다." -ForegroundColor Yellow
    $confirm = Read-Host "   삭제하고 새로 생성하시겠습니까? (y/N)"

    if ($confirm -eq "y" -or $confirm -eq "Y") {
        Write-Host "   기존 가상환경 삭제 중..."
        Remove-Item -Recurse -Force venv
        Write-Host "   ✓ 삭제 완료" -ForegroundColor Green
    }
    else {
        Write-Host "   설치를 중단합니다." -ForegroundColor Yellow
        exit 0
    }
}

# ----------------------------------------------------------------------------
# 3단계: 가상환경 생성
# ----------------------------------------------------------------------------
Write-Host ""
Write-Host "3. 가상환경 생성 중..."
python -m venv venv
Write-Host "   ✓ venv 가상환경이 생성되었습니다." -ForegroundColor Green

# ----------------------------------------------------------------------------
# 4단계: 가상환경 활성화 및 pip 업그레이드
# ----------------------------------------------------------------------------
Write-Host ""
Write-Host "4. 가상환경 활성화 및 pip 업그레이드 중..."

# 가상환경 활성화
& ".\venv\Scripts\Activate.ps1"

# pip 업그레이드
python -m pip install --upgrade pip setuptools wheel --quiet
Write-Host "   ✓ pip 업그레이드 완료" -ForegroundColor Green

# ----------------------------------------------------------------------------
# 5단계: requirements.txt 설치
# ----------------------------------------------------------------------------
Write-Host ""
Write-Host "5. 필요한 패키지 설치 중..."
Write-Host "   (이 과정은 몇 분 정도 걸릴 수 있습니다...)" -ForegroundColor Yellow

if (Test-Path "requirements.txt") {
    pip install -r requirements.txt
    Write-Host "   ✓ 모든 패키지 설치 완료" -ForegroundColor Green
}
else {
    Write-Host "   ⚠️  requirements.txt 파일을 찾을 수 없습니다." -ForegroundColor Red
    exit 1
}

# ----------------------------------------------------------------------------
# 6단계: 설치 확인
# ----------------------------------------------------------------------------
Write-Host ""
Write-Host "6. 설치 확인 중..."

try {
    python -c "import torch; import torchvision; print(f'   ✓ PyTorch {torch.__version__} 설치됨')"
    python -c "import numpy; import pandas; print('   ✓ NumPy, Pandas 설치됨')"
    python -c "import matplotlib; import seaborn; print('   ✓ Matplotlib, Seaborn 설치됨')"
}
catch {
    Write-Host "   ⚠️  일부 패키지 확인 실패" -ForegroundColor Yellow
}

# ----------------------------------------------------------------------------
# 완료 메시지
# ----------------------------------------------------------------------------
Write-Host ""
Write-Host "======================================================================"
Write-Host "✓ 가상환경 설정이 완료되었습니다!" -ForegroundColor Green
Write-Host "======================================================================"
Write-Host ""
Write-Host "다음 명령어로 가상환경을 활성화하세요:"
Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor Cyan
Write-Host ""
Write-Host "가상환경을 비활성화하려면:"
Write-Host "  deactivate" -ForegroundColor Cyan
Write-Host ""
Write-Host "이제 다음 단계를 진행하세요:"
Write-Host "  python download_dataset.py" -ForegroundColor Cyan
Write-Host "======================================================================"
