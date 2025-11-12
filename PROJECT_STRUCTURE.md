# 프로젝트 구조 가이드

이 문서는 Food-101 Image Classification 프로젝트의 폴더 구조와 각 파일의 역할을 설명합니다.

## 전체 구조 개요

```
llm_prj/
├── README.md                    # 프로젝트 메인 문서
├── PROJECT_STRUCTURE.md         # 본 문서 (구조 가이드)
├── requirements.txt             # Python 패키지 의존성
│
├── data/                        # 데이터셋 디렉토리
├── notebooks/                   # Jupyter 노트북
├── scripts/                     # 유틸리티 스크립트
├── docs/                        # 문서 및 발표 자료
├── outputs/                     # 훈련 결과물
└── archive/                     # 이전 버전 백업
```

---

## 상세 구조

### 1. 루트 디렉토리

```
/root/llm_prj/
├── README.md                    # 프로젝트 소개 및 사용법
├── PROJECT_STRUCTURE.md         # 프로젝트 구조 가이드 (본 문서)
├── requirements.txt             # pip install 용 패키지 목록
└── .gitignore                   # Git 무시 파일 목록
```

**주요 파일:**
- `README.md`: 프로젝트 시작점. 설치, 사용법, 주요 기능 설명
- `requirements.txt`: 프로젝트에 필요한 Python 패키지 목록

---

### 2. data/ - 데이터셋 디렉토리

```
data/
├── dataset_path.txt             # 데이터셋 경로 정보
└── food-101/                    # Food-101 데이터셋 (다운로드 후 생성)
    └── versions/1/food-101/food-101/
        ├── images/              # 음식 이미지 (101개 카테고리)
        │   ├── apple_pie/       # 클래스별 폴더
        │   ├── hamburger/
        │   └── ...
        └── meta/                # 메타데이터
            ├── classes.txt      # 클래스 목록
            ├── train.txt        # 훈련 이미지 목록
            └── test.txt         # 테스트 이미지 목록
```

**역할:**
- 데이터셋 저장 및 관리
- `dataset_path.txt`는 노트북에서 데이터 경로를 읽는데 사용

**용량:** 약 5GB

---

### 3. notebooks/ - Jupyter 노트북

```
notebooks/
└── food101_training.ipynb       # 메인 훈련 노트북 (GPU 최적화 버전)
```

**food101_training.ipynb 구조:**
1. 환경 설정 및 라이브러리 임포트
2. 데이터 로딩 및 전처리
3. 모델 구성 (ResNet18)
4. 하이퍼파라미터 탐색 (5개 실험 자동 수행)
5. 결과 시각화 및 평가

**특징:**
- GPU 최적화 (배치 크기 256, Mixed Precision)
- 자동 하이퍼파라미터 탐색
- 최고 성능 모델 자동 선택

---

### 4. scripts/ - 유틸리티 스크립트

```
scripts/
├── download_dataset.py          # Food-101 데이터셋 다운로드
├── explore_dataset.py           # 데이터셋 구조 탐색 및 통계
│
├── setup_venv.sh                # Linux/Mac 가상환경 설정
├── setup_venv.ps1               # Windows 가상환경 설정
├── install_miniconda.sh         # Miniconda 설치 스크립트
└── setup_conda.sh               # Conda 환경 자동 설정
```

**각 스크립트 설명:**

#### download_dataset.py
- Kaggle API를 통해 Food-101 데이터셋 자동 다운로드
- `dataset_path.txt` 자동 생성

**사용법:**
```bash
python scripts/download_dataset.py
```

#### explore_dataset.py
- 데이터셋 구조 분석
- 클래스 분포, 이미지 크기 통계 출력

**사용법:**
```bash
python scripts/explore_dataset.py
```

#### setup_venv.sh / setup_venv.ps1
- Python 가상환경 자동 생성 및 패키지 설치
- 플랫폼별 스크립트 (Linux/Mac, Windows)

#### setup_conda.sh
- Conda 환경 자동 생성 및 PyTorch 설치
- GPU/CPU 자동 감지

---

### 5. docs/ - 문서 및 발표 자료

```
docs/
├── README.md                    # 프로젝트 문서 사본
├── presentation_script.txt      # 발표 스크립트 (10-12분 분량)
└── qna_material.txt             # Q&A 대응 자료 (40+ 질문)
```

**문서 설명:**

#### presentation_script.txt
- 13개 슬라이드 구성 발표 스크립트
- 프로젝트 소개, 기술 설명, 결과 분석 포함
- 발표 시간 배분 가이드 포함

**주요 내용:**
- 슬라이드 1-2: 프로젝트 소개
- 슬라이드 3-6: 데이터셋 및 모델
- 슬라이드 7-10: 훈련 및 결과
- 슬라이드 11-13: 향후 개선 및 결론

#### qna_material.txt
- 7개 카테고리, 40개 이상 예상 질문과 답변
- 기술적 깊이에 따른 기본/심화 답변 제공

**카테고리:**
1. 프로젝트 일반
2. 데이터셋 관련
3. 모델 및 아키텍처
4. 훈련 과정
5. 결과 및 평가
6. 기술적 세부사항
7. 향후 계획

---

### 6. outputs/ - 훈련 결과물

```
outputs/
├── models/                      # 학습된 모델 파일
│   ├── best_model.pth           # 최고 성능 모델
│   └── best_model_refactored.pth
│
└── images/                      # 결과 시각화
    ├── training_results.png     # 학습 곡선 그래프
    ├── sample_predictions.png   # 예측 결과 샘플
    └── ...
```

**역할:**
- 훈련 중 생성되는 모델 가중치 저장
- 시각화 결과 이미지 저장
- 재학습 없이 모델 재사용 가능

**용량:** 약 200MB (모델 파일 포함 시)

---

### 7. archive/ - 이전 버전 백업

```
archive/
├── food101_complete.ipynb       # 이전 완전 버전
├── food101_eda.ipynb            # EDA 노트북
├── food101_optimized.ipynb      # 최적화 이전 버전
├── food101_optimized.ipynb.bak  # 백업 파일
├── food101_refactored_script.py # 스크립트 버전
├── presentation_script.md       # 마크다운 발표 자료
└── qna.md                       # 마크다운 Q&A
```

**역할:**
- 프로젝트 이전 버전 보관
- 개발 히스토리 추적
- 필요 시 이전 버전 복원 가능

**주의:** 일반적으로 사용하지 않음. 참고용으로만 보관.

---

## 파일 크기 요약

| 디렉토리 | 예상 크기 | 설명 |
|---------|----------|------|
| `data/` | ~5GB | Food-101 데이터셋 |
| `outputs/models/` | ~200MB | 학습된 모델 |
| `outputs/images/` | ~10MB | 시각화 이미지 |
| `notebooks/` | ~1MB | Jupyter 노트북 |
| `scripts/` | <1MB | Python/Shell 스크립트 |
| `docs/` | <1MB | 문서 파일 |
| `archive/` | ~10MB | 이전 버전 백업 |

**총 용량:** 약 5.2GB

---

## 주요 파일 흐름도

```
1. 초기 설정
   scripts/setup_conda.sh → requirements.txt
   ↓

2. 데이터 다운로드
   scripts/download_dataset.py → data/food-101/
   ↓

3. 훈련 실행
   notebooks/food101_training.ipynb
   - 데이터 로드 (data/dataset_path.txt 참조)
   - 모델 훈련
   ↓

4. 결과 저장
   - outputs/models/best_model.pth
   - outputs/images/*.png
   ↓

5. 발표 준비
   docs/presentation_script.txt
   docs/qna_material.txt
```

---

## 사용 시나리오별 가이드

### 시나리오 1: 처음 프로젝트 시작
1. `README.md` 읽기
2. `scripts/setup_conda.sh` 실행 (환경 설정)
3. `scripts/download_dataset.py` 실행 (데이터 다운로드)
4. `notebooks/food101_training.ipynb` 열기 및 실행

### 시나리오 2: 모델만 다시 학습
1. `notebooks/food101_training.ipynb` 열기
2. Cell 4.2에서 하이퍼파라미터 조정
3. Cell 4.3 실행 (훈련 시작)

### 시나리오 3: 발표 준비
1. `docs/presentation_script.txt` 읽고 발표 연습
2. `docs/qna_material.txt`에서 예상 질문 숙지
3. `outputs/images/`의 시각화 자료 활용

### 시나리오 4: 이전 버전 확인
1. `archive/` 폴더 확인
2. 필요한 노트북 열기

---

## 파일 생성 및 삭제 가이드

### 자동 생성 파일
다음 파일들은 스크립트/노트북 실행 시 자동으로 생성됩니다:
- `data/dataset_path.txt` (download_dataset.py)
- `outputs/models/*.pth` (training notebook)
- `outputs/images/*.png` (training notebook)

### 삭제 가능 파일
디스크 공간 확보를 위해 삭제 가능한 파일:
- `archive/` 전체 폴더 (이전 버전 불필요 시)
- `outputs/models/*.pth` (재학습 시)
- `outputs/images/*.png` (재생성 가능)

### 절대 삭제 금지 파일
- `notebooks/food101_training.ipynb` (메인 노트북)
- `data/food-101/` (재다운로드 시간 소요)
- `scripts/` 폴더 (재설정 필요)
- `README.md`, `requirements.txt`

---

## 추가 참고사항

### 경로 참조 방법
노트북에서 파일 참조 시 상대 경로 사용:
```python
# 좋은 예 (상대 경로)
DATA_ROOT = Path("../data/food-101/food-101")

# 나쁜 예 (절대 경로)
DATA_ROOT = Path("/root/llm_prj/data/food-101")
```

### Git 관리
`.gitignore`에 포함된 디렉토리:
- `data/food-101/` (데이터셋)
- `outputs/models/` (대용량 모델)
- `archive/` (백업)
- `.ipynb_checkpoints/` (임시 파일)

### 새 파일 추가 시
새 노트북이나 스크립트 추가 시:
- 적절한 폴더에 배치
- 본 문서(PROJECT_STRUCTURE.md) 업데이트
- README.md에 간단히 언급

---

## 문의

프로젝트 구조 관련 질문이 있으시면 이슈를 생성해주세요.
