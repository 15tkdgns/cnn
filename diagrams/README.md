# 📊 Diagrams - 프로젝트 시각화 다이어그램

이 폴더에는 Food-101 프로젝트의 흐름과 구조를 시각화한 다이어그램들이 포함되어 있습니다.

## 📁 포함된 파일

### 1. MERMAID_GUIDE.md
**Mermaid 다이어그램을 보는 방법 가이드**

- GitHub에서 보기 (가장 쉬움)
- VS Code에서 보기
- 온라인 에디터 (mermaid.live)
- 모바일에서 보기
- 이미지로 변환하기
- 트러블슈팅 및 FAQ

👉 **[가이드 보기](MERMAID_GUIDE.md)**

---

### 2. PROJECT_FLOW_MERMAID.md
**10가지 Mermaid 플로우차트 모음**

다음 다이어그램이 포함되어 있습니다:

1. 전체 시스템 아키텍처 (graph)
2. 모델 훈련 파이프라인 (flowchart)
3. API 서버 시작 과정 (flowchart)
4. 이미지 분류 요청 처리 (sequence diagram)
5. Grad-CAM 생성 과정 (flowchart)
6. YOLO 객체 탐지 (flowchart)
7. 프론트엔드 상호작용 (state diagram)
8. 데이터 변환 체인 - 훈련 시 (flowchart)
9. 데이터 변환 체인 - 추론 시 (flowchart)
10. 전체 요청-응답 시퀀스 (sequence diagram)
11. 성능 최적화 포인트 (mindmap)

👉 **[다이어그램 보기](PROJECT_FLOW_MERMAID.md)**

---

### 3. PROJECT_FLOW_DIAGRAM.md
**텍스트 기반 상세 흐름도**

- ASCII 아트 스타일
- 모든 터미널/에디터에서 볼 수 있음
- 6가지 주요 섹션 포함
- 처리 시간 및 메모리 사용량 정보

👉 **[흐름도 보기](PROJECT_FLOW_DIAGRAM.md)**

---

### 4. mermaid_example.md
**Mermaid 예제 모음**

다양한 다이어그램 타입 예제:
- 플로우차트
- 시퀀스 다이어그램
- 클래스 다이어그램
- 간트 차트
- 파이 차트
- 상태 다이어그램
- 마인드맵

👉 **[예제 보기](mermaid_example.md)**

---

## 🚀 빠른 시작

### 가장 쉬운 방법: GitHub에서 보기

1. **메인 플로우차트**
   ```
   https://github.com/15tkdgns/cnn/blob/main/diagrams/PROJECT_FLOW_MERMAID.md
   ```

2. **예제 모음**
   ```
   https://github.com/15tkdgns/cnn/blob/main/diagrams/mermaid_example.md
   ```

3. **가이드**
   ```
   https://github.com/15tkdgns/cnn/blob/main/diagrams/MERMAID_GUIDE.md
   ```

**모든 Mermaid 다이어그램이 자동으로 렌더링됩니다!**

---

## 💻 로컬에서 보기

### VS Code 사용

```bash
# 1. Mermaid 확장 프로그램 설치
code --install-extension bierner.markdown-mermaid

# 2. 파일 열기
code diagrams/PROJECT_FLOW_MERMAID.md

# 3. 미리보기
# Windows/Linux: Ctrl + Shift + V
# Mac: Cmd + Shift + V
```

### 온라인 에디터 사용

```
1. https://mermaid.live 접속
2. 다이어그램 코드 복사
3. 에디터에 붙여넣기
4. 실시간 렌더링 확인
```

---

## 📖 다이어그램 내용 요약

### 시스템 흐름
- **훈련 단계**: 데이터 다운로드 → 전처리 → 모델 훈련 → 모델 저장
- **서비스 단계**: 모델 로드 → API 서버 시작 → 요청 처리 → 결과 반환

### 주요 컴포넌트
- **데이터**: Food-101 (101,000 이미지, 101 클래스)
- **모델**: ResNet18 (전이학습, 76.32% 정확도)
- **백엔드**: FastAPI + PyTorch + GPU 최적화
- **프론트엔드**: React + ChatGPT 스타일 UI
- **추가 기능**: Grad-CAM, YOLO 객체 탐지

### 데이터 흐름
```
사용자 → React → FastAPI → PyTorch → GPU → 결과
```

---

## 🎨 다이어그램 타입별 설명

### Flowchart (플로우차트)
순차적인 프로세스를 표현
- 모델 훈련 파이프라인
- API 서버 시작 과정
- 데이터 변환 체인

### Sequence Diagram (시퀀스 다이어그램)
컴포넌트 간 통신 흐름을 표현
- 이미지 분류 요청 처리
- 전체 요청-응답 시퀀스

### State Diagram (상태 다이어그램)
상태 전환을 표현
- 프론트엔드 상호작용

### Graph (그래프)
전체 시스템 구조를 표현
- 시스템 아키텍처

### Mindmap (마인드맵)
계층적 정보를 표현
- 성능 최적화 포인트

---

## 🔧 활용 방법

### 발표 자료 준비
1. GitHub에서 다이어그램 확인
2. mermaid.live에서 PNG 다운로드
3. PowerPoint/Keynote에 삽입

### 문서 작성
1. VS Code에서 실시간 미리보기로 작성
2. GitHub에 push하면 자동 렌더링

### 팀 공유
1. GitHub 링크 공유
2. 또는 PNG 이미지 첨부

---

## 📚 추가 리소스

### 공식 문서
- [Mermaid 공식 문서](https://mermaid.js.org/)
- [GitHub Mermaid 지원](https://github.blog/2022-02-14-include-diagrams-markdown-files-mermaid/)

### 온라인 도구
- [Mermaid Live Editor](https://mermaid.live)
- [Mermaid Cheat Sheet](https://jojozhuang.github.io/tutorial/mermaid-cheat-sheet/)

---

## ❓ 자주 묻는 질문

**Q: 다이어그램이 렌더링되지 않아요.**
A: MERMAID_GUIDE.md의 트러블슈팅 섹션을 참조하세요.

**Q: 다이어그램을 수정하고 싶어요.**
A: VS Code에서 파일을 열고 편집한 후 커밋하세요. GitHub에서 자동으로 렌더링됩니다.

**Q: 이미지로 저장하려면?**
A: mermaid.live에서 PNG/SVG로 다운로드할 수 있습니다.

---

**즐거운 다이어그램 탐험 되세요! 🚀**

← [프로젝트 루트로 돌아가기](../)
