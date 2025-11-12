# Food-101 이미지 분류 프로젝트

Food-101 데이터셋을 활용한 음식 이미지 분류 프로젝트입니다. PyTorch를 사용하여 101가지 음식 카테고리를 분류하는 모델을 학습합니다.

## 프로젝트 구조

```
llm_prj/
├── README.md                    # 프로젝트 문서 (본 파일)
├── .gitignore                   # Git 무시 파일 목록
├── requirements.txt             # 필요한 Python 패키지 목록
├── setup_venv.sh                # Linux/Mac 가상환경 자동 설정 스크립트
├── setup_venv.ps1               # Windows 가상환경 자동 설정 스크립트
├── download_dataset.py          # Food-101 데이터셋 다운로드 스크립트
├── explore_dataset.py           # 데이터셋 구조 탐색 스크립트
├── food101_eda.ipynb            # EDA 및 시각화 노트북
├── dataset_path.txt             # 데이터셋 경로 (자동 생성)
└── dataset_summary.json         # 데이터셋 요약 정보 (자동 생성)
```

## 데이터셋 정보

- **이름**: Food-101
- **출처**: Kaggle (dansbecker/food-101)
- **총 클래스**: 101개 음식 카테고리
- **총 이미지**: 101,000장
  - 훈련 이미지: 75,750장 (클래스당 750장)
  - 테스트 이미지: 25,250장 (클래스당 250장)
- **이미지 크기**: 최대 변 길이 512픽셀

## 빠른 시작

### 0. 가상환경 설정 (필수)

프로젝트 시작 전 반드시 가상환경을 설정하세요. 가상환경을 사용하면 프로젝트별로 독립된 패키지 환경을 유지할 수 있습니다.

#### 🚀 빠른 설정 (자동 스크립트)

**Linux/Mac:**
```bash
chmod +x setup_venv.sh
./setup_venv.sh
```

**Windows (PowerShell):**
```powershell
# 실행 정책 설정 (처음 한 번만)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 스크립트 실행
.\setup_venv.ps1
```

#### 옵션 A: venv 사용 (Python 내장, 수동 설정)

```bash
# 1. 가상환경 생성
python -m venv venv

# 2. 가상환경 활성화
# Linux/Mac:
source venv/bin/activate

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Windows (CMD):
venv\Scripts\activate.bat

# 3. 활성화 확인 (프롬프트 앞에 (venv) 표시됨)
# (venv) $
```

#### 옵션 B: conda 사용 (딥러닝 프로젝트에 추천)

```bash
# 1. conda 환경 생성 (Python 3.10 사용)
conda create -n food101 python=3.10 -y

# 2. 환경 활성화
conda activate food101

# 3. PyTorch 설치 (CUDA 지원)
# GPU 있는 경우:
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y

# CPU만 있는 경우:
conda install pytorch torchvision cpuonly -c pytorch -y

# 4. 활성화 확인 (프롬프트 앞에 (food101) 표시됨)
# (food101) $
```

**가상환경 비활성화:**
```bash
# venv:
deactivate

# conda:
conda deactivate
```

### 1. 패키지 설치

가상환경을 활성화한 상태에서 필요한 패키지를 설치합니다.

```bash
# venv 사용 시:
pip install -r requirements.txt

# conda 사용 시 (PyTorch 제외):
pip install kagglehub pandas matplotlib seaborn tqdm jupyter ipykernel ipywidgets
```

### 2. 데이터셋 다운로드

```bash
# Food-101 데이터셋 다운로드 (약 10GB)
python download_dataset.py
```

### 3. 데이터 탐색

**옵션 A: Python 스크립트**
```bash
python explore_dataset.py
```

**옵션 B: Jupyter 노트북**
```bash
jupyter notebook food101_eda.ipynb
```

## 주요 기능

- **데이터셋 다운로드**: Kaggle API를 통한 자동 다운로드
- **EDA**: 클래스 분포, 이미지 크기, 샘플 시각화
- **데이터 분석**: 훈련/테스트 분할 분석, 이미지 특성 분석

---

## CNN 기반 음식 이미지 분류 프로젝트 주의사항

본 문서는 CNN 기반 음식 이미지 분류(예: 김밥, 햄버거, 피자) 프로젝트 진행 시 주니어 개발자가 흔히 범하는 실수와 이를 방지하기 위한 구체적인 주의사항을 기술합니다.

⛔️ '바이브 코딩' 함정 (챗봇이 자주 하는 실수) 피하기
'느낌'에 의존한 코딩은 시간을 아껴주는 것 같지만, 디버깅 지옥으로 가는 지름길이 될 수 있습니다. 챗봇도 자주 하는 다음 실수들을 피하십시오.

"경로 하드코딩의 저주"

오류 예시: load_image("C:/Users/MyPC/Desktop/project/data/...")

왜 문제인가: 팀원의 PC나 금요일에 연결할 서버(FastAPI)에는 MyPC 폴더가 없습니다. 코드는 당신의 컴퓨터에서만 작동합니다.

해결: 항상 상대 경로(./data/train)를 사용하십시오.

"복붙 후 임포트(Import) 누락"

오류 예시: np.array(...) 코드를 복붙하고 import numpy as np를 빼먹어 NameError 발생.

왜 문제인가: 코드는 필요한 도구(라이브러리)를 선언해야만 사용할 수 있습니다.

해결: 필요한 라이브러리(tensorflow, numpy, matplotlib 등)는 코드 최상단에 모아서 선언하십시오.

"데이터 차원(Shape) 불일치"

오류 예시: 모델은 (224, 224, 3) 이미지를 기다리는데, 전처리를 깜빡하고 (400, 300, 3) 이미지를 그냥 주입.

왜 문제인가: CNN 모델은 약속된 규격의 입력만 받습니다.

해결: 모델의 input_shape과 이미지 리사이징 크기를 반드시 일치시키십시오.

"검증 데이터로 학습시키는 행위 (데이터 누수)"

오류 예시: 실수로 검증용(validation) 데이터를 학습용(train) 데이터에 섞어 넣음.

왜 문제인가: 모델이 '정답(검증 데이터)'을 미리 외워버려 정확도가 99%처럼 보입니다. 이는 속임수이며, 새로운 데이터를 만나면 성능이 급락합니다.

해결: train 폴더와 val 폴더를 명확히 분리하고, 학습 시 validation_data 인자에 검증 데이터셋만 정확히 전달하십시오.

1. 데이터 준비 단계
과도한 데이터 수집 지양: 완벽한 데이터를 찾기 위해 시간을 허비하지 마십시오. 클래스당 100~200장의 이미지만으로도 초기 모델 학습에 충분합니다. 구글 이미지 크롤링, Kaggle 등을 활용하여 빠르게 데이터를 확보하십시오. 약간의 노이즈는 허용됩니다.

클래스 불균형 방지: 각 클래스(음식 종류)별 이미지 개수를 비슷하게 유지해야 합니다. 특정 클래스의 데이터가 너무 많거나 적으면 모델이 편향되게 학습될 수 있습니다.

데이터 분할 필수: 전체 데이터를 학습용(Train)과 검증용(Validation)으로 반드시 분리하십시오(예: 8:2 비율). 검증 데이터 없이 학습 데이터만으로 성능을 평가하면 과적합 여부를 파악할 수 없습니다.

2. 모델 구현 및 학습 단계
입력 이미지 크기 일치: (위의 '바이브 코딩' 섹션 참고) 모델의 입력 레이어가 기대하는 크기(예: 224x224)와 실제 데이터의 크기를 반드시 일치시켜야 합니다. 이미지 로드 시 리사이징 전처리를 필수적으로 포함하십시오.

경로 설정 주의: (위의 '바이브 코딩' 섹션 참고) 데이터 로드 시 절대 경로 대신 상대 경로를 사용하여 코드의 이식성을 높이십시오.

전이 학습(Transfer Learning) 적극 활용: 80% 이상의 정확도 목표 달성을 위해 처음부터 모델을 설계하기보다 사전 학습된 모델(MobileNetV2, ResNet50 등)을 활용하는 것이 효율적입니다.

과적합(Overfitting) 모니터링: 학습 정확도는 높지만 검증 정확도가 낮다면 과적합을 의심해야 합니다. 이를 해결하기 위해 데이터 증강(Data Augmentation), 드롭아웃(Dropout) 추가, 모델 복잡도 감소 등의 기법을 적용하십시오.

3. 평가 및 시각화 단계
별도 평가 코드 작성: 학습 과정에서 제공되는 지표 외에 Confusion Matrix와 같은 심층적인 분석 도구는 별도의 코드로 구현해야 합니다.

발표 자료 준비: 단순한 정확도 수치뿐만 아니라 학습 곡선(Loss/Accuracy 그래프), Confusion Matrix, 실제 예측 결과 예시 등을 포함하여 시각적으로 풍부한 발표 자료를 준비하십시오.

4. 기타
버전 관리: 코드 변경 사항을 지속적으로 추적하고 관리하기 위해 Git을 사용하십시오.

모델 저장: FastAPI 연동을 위해 학습이 완료된 모델을 파일(.h5, .keras 또는 .pt 형식)로 저장해 두십시오. model.save()를 잊지 마십시오.