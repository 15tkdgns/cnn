# Mermaid 다이어그램 예제

이 파일은 Mermaid 다이어그램이 어떻게 보이는지 테스트하기 위한 간단한 예제입니다.

## 예제 1: 간단한 플로우차트

```mermaid
flowchart TD
    Start([시작]) --> Input[이미지 업로드]
    Input --> Process[AI 처리]
    Process --> Result{성공?}
    Result -->|Yes| Success[결과 표시]
    Result -->|No| Error[에러 표시]
    Success --> End([종료])
    Error --> End

    style Start fill:#e1f5ff
    style Success fill:#e1ffe1
    style Error fill:#ffe1e1
    style End fill:#f0e1ff
```

## 예제 2: 시퀀스 다이어그램

```mermaid
sequenceDiagram
    participant User as 사용자
    participant Frontend as React 앱
    participant Backend as FastAPI
    participant Model as ResNet18

    User->>Frontend: 이미지 업로드
    Frontend->>Backend: POST /predict
    Backend->>Model: 추론 요청
    Model->>Backend: 결과 반환
    Backend->>Frontend: JSON 응답
    Frontend->>User: 결과 표시
```

## 예제 3: 클래스 다이어그램

```mermaid
classDiagram
    class Food101Classifier {
        +model: ResNet18
        +device: str
        +classes: List
        +predict(image)
        +load_model()
    }

    class FastAPIServer {
        +app: FastAPI
        +classifier: Food101Classifier
        +predict_endpoint()
        +health_check()
    }

    class ReactApp {
        +image: File
        +result: Dict
        +handleUpload()
        +displayResult()
    }

    FastAPIServer --> Food101Classifier
    ReactApp --> FastAPIServer : HTTP
```

## 예제 4: 간트 차트 (프로젝트 타임라인)

```mermaid
gantt
    title Food-101 프로젝트 개발 타임라인
    dateFormat  YYYY-MM-DD
    section 데이터 준비
    데이터셋 다운로드      :done, 2024-11-09, 1d
    데이터 전처리          :done, 2024-11-09, 1d
    section 모델 개발
    모델 설계             :done, 2024-11-10, 1d
    하이퍼파라미터 튜닝    :done, 2024-11-10, 2d
    모델 훈련             :done, 2024-11-10, 1d
    section 백엔드 개발
    FastAPI 구현          :done, 2024-11-11, 1d
    API 테스트            :done, 2024-11-11, 1d
    section 프론트엔드
    React 앱 개발         :done, 2024-11-11, 1d
    UI/UX 개선            :done, 2024-11-11, 1d
    section 배포
    문서화                :done, 2024-11-12, 1d
    GitHub 연동           :active, 2024-11-12, 1d
```

## 예제 5: 파이 차트

```mermaid
pie title 프로젝트 구성 요소 비중
    "Backend (Python)" : 40
    "Frontend (React)" : 30
    "Model Training" : 20
    "Documentation" : 10
```

## 예제 6: 상태 다이어그램

```mermaid
stateDiagram-v2
    [*] --> 대기중
    대기중 --> 이미지선택 : 사용자 업로드
    이미지선택 --> 검증 : 파일 확인
    검증 --> 에러 : 유효하지 않음
    검증 --> 업로드완료 : 유효함
    업로드완료 --> 분석중 : 분석 시작
    분석중 --> 완료 : 성공
    분석중 --> 에러 : 실패
    에러 --> 대기중 : 재시도
    완료 --> 대기중 : 새 분석
```

## 예제 7: 마인드맵

```mermaid
mindmap
  root((Food-101 프로젝트))
    데이터
      Food-101 Dataset
      101 클래스
      101,000 이미지
    모델
      ResNet18
      전이학습
      76.32% 정확도
    백엔드
      FastAPI
      PyTorch
      GPU 최적화
    프론트엔드
      React
      ChatGPT 스타일
      실시간 분석
    기능
      이미지 분류
      Grad-CAM
      YOLO 탐지
```

## 이 파일을 보는 방법

### 1. GitHub에서 보기 (추천)
```
https://github.com/15tkdgns/cnn/blob/main/mermaid_example.md
```
→ 자동으로 렌더링됩니다!

### 2. VS Code에서 보기
```bash
# 1. Mermaid 확장 프로그램 설치
code --install-extension bierner.markdown-mermaid

# 2. 파일 열고 미리보기
code mermaid_example.md
# Ctrl+Shift+V 눌러서 미리보기
```

### 3. 온라인에서 보기
1. https://mermaid.live 접속
2. 위 코드 복사 & 붙여넣기
3. 실시간 렌더링 확인!

---

**참고:** 모든 다이어그램이 즉시 렌더링되는지 확인하려면 GitHub에서 이 파일을 열어보세요!
