/*
 * Food Classifier - React Frontend
 *
 * 주요 기능:
 * - 음식 이미지 분류 (ResNet18, 101 classes)
 * - Grad-CAM 히트맵 시각화 (AI 판단 근거)
 * - YOLO 객체 탐지 (80 classes)
 * - 드래그 앤 드롭, 붙여넣기 지원
 */

import React, { useState, useRef } from 'react';
import './App.css';
import axios from 'axios';

function App() {
  // ===== State 관리 =====
  // 업로드된 이미지 파일 객체
  const [image, setImage] = useState(null);
  // 이미지 미리보기 URL (base64 Data URI)
  const [preview, setPreview] = useState(null);
  // API 응답 결과 (prediction, top5, gradcam, detections 등)
  const [result, setResult] = useState(null);
  // 로딩 상태 (API 요청 중)
  const [loading, setLoading] = useState(false);
  // 에러 메시지
  const [error, setError] = useState(null);
  // 드래그 앤 드롭 상태 (UI 피드백용)
  const [isDragging, setIsDragging] = useState(false);
  // Grad-CAM 활성화 여부 (음식 분류 모드에서만 사용)
  const [showGradCAM, setShowGradCAM] = useState(false);
  // 작동 모드: 'classify' (음식 분류) 또는 'detect' (객체 탐지)
  const [mode, setMode] = useState('classify');
  // 파일 입력 요소 참조 (프로그래매틱 클릭용)
  const fileInputRef = useRef(null);

  // 백엔드 API URL (환경 변수 또는 기본값)
  const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

  // ===== 이벤트 핸들러 =====

  /**
   * 파일 선택 처리
   * 파일 입력, 드래그 앤 드롭, 붙여넣기에서 공통으로 사용
   *
   * @param {File} file - 선택된 파일 객체
   *
   * 동작:
   * 1. 파일 유효성 검사 (이미지 파일인지 확인)
   * 2. State 업데이트 (image, result, error 초기화)
   * 3. FileReader로 base64 Data URI 생성 (미리보기용)
   */
  const handleFileSelect = (file) => {
    if (!file) return;

    // 이미지 파일인지 검사 (MIME 타입 확인)
    if (!file.type.startsWith('image/')) {
      setError('이미지 파일만 업로드 가능합니다.');
      return;
    }

    // State 업데이트
    setImage(file);        // 업로드할 파일 저장
    setResult(null);       // 이전 결과 초기화
    setError(null);        // 에러 메시지 초기화

    // FileReader로 미리보기 이미지 생성
    // readAsDataURL: 파일을 base64 인코딩된 Data URI로 변환
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target.result);  // Data URI 저장
    reader.readAsDataURL(file);
  };

  /**
   * 드래그 앤 드롭 처리
   * 파일을 드롭 영역에 놓았을 때 호출
   */
  const handleDrop = (e) => {
    e.preventDefault();  // 기본 동작 방지 (브라우저가 파일을 여는 것 방지)
    setIsDragging(false);  // 드래그 상태 해제

    const file = e.dataTransfer.files[0];  // 첫 번째 파일만 사용
    handleFileSelect(file);
  };

  /**
   * 드래그 오버 처리
   * 파일을 드래그하여 드롭 영역 위에 있을 때 호출
   */
  const handleDragOver = (e) => {
    e.preventDefault();  // 기본 동작 방지 (드롭 허용을 위해 필수)
    setIsDragging(true);  // 드래그 상태 활성화 (UI 피드백)
  };

  /**
   * 드래그 리브 처리
   * 파일을 드래그하여 드롭 영역을 벗어났을 때 호출
   */
  const handleDragLeave = () => {
    setIsDragging(false);  // 드래그 상태 해제
  };

  /**
   * 붙여넣기 처리 (Ctrl+V)
   * 클립보드의 이미지를 붙여넣을 때 호출
   */
  const handlePaste = (e) => {
    // 클립보드 아이템들을 순회
    const items = e.clipboardData.items;
    for (let item of items) {
      // 이미지 타입 아이템 찾기
      if (item.type.indexOf('image') !== -1) {
        const file = item.getAsFile();  // File 객체로 변환
        handleFileSelect(file);
        break;  // 첫 번째 이미지만 사용
      }
    }
  };

  /**
   * 예측 수행 (API 요청)
   * "분석하기" 버튼 클릭 시 호출
   *
   * 동작:
   * 1. 현재 모드에 따라 API 엔드포인트 결정
   *    - classify + Grad-CAM: /predict/gradcam
   *    - classify: /predict
   *    - detect: /detect
   * 2. FormData로 이미지 파일 전송 (multipart/form-data)
   * 3. 응답 데이터를 result state에 저장
   * 4. 에러 처리 및 로딩 상태 관리
   */
  const handlePredict = async () => {
    if (!image) return;

    setLoading(true);  // 로딩 시작
    setError(null);    // 이전 에러 초기화

    try {
      // FormData 생성 (multipart/form-data 형식)
      const formData = new FormData();
      formData.append('file', image);

      // 모드에 따라 API 엔드포인트 결정
      let endpoint;
      if (mode === 'detect') {
        // YOLO 객체 탐지 모드
        endpoint = '/detect';
      } else {
        // 음식 분류 모드 (Grad-CAM 옵션)
        endpoint = showGradCAM ? '/predict/gradcam' : '/predict';
      }

      // API 요청 (POST)
      const response = await axios.post(`${API_URL}${endpoint}`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      // 응답 데이터 저장
      setResult(response.data);
    } catch (err) {
      // 에러 처리 (서버 에러 메시지 또는 기본 메시지)
      setError(err.response?.data?.detail || '예측에 실패했습니다. 다시 시도해주세요.');
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);  // 로딩 종료
    }
  };

  /**
   * 초기화 처리
   * "다시 선택" 또는 "새로운 이미지 분석" 버튼 클릭 시 호출
   *
   * 동작:
   * 1. 모든 state 초기화
   * 2. 파일 입력 요소 초기화
   */
  const handleReset = () => {
    setImage(null);
    setPreview(null);
    setResult(null);
    setError(null);
    // 파일 입력 요소 초기화 (같은 파일 재선택 가능하도록)
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // ===== JSX 렌더링 =====
  return (
    <div className="app" onPaste={handlePaste}>
      {/* 헤더: 애플리케이션 제목 */}
      <header className="header">
        <div className="header-content">
          <h1 className="logo">Food Classifier</h1>
        </div>
      </header>

      {/* 메인 콘텐츠 */}
      <main className="main-content">
        <div className="container">

          {/* 초기 화면: 이미지가 업로드되지 않은 상태 */}
          {!preview && (
            <div className="welcome-section">
              <div className="welcome-icon">[ Image ]</div>
              <h2 className="welcome-title">음식 이미지를 분석해드립니다</h2>
              <p className="welcome-subtitle">
                이미지를 업로드하거나 붙여넣기하여 101가지 음식 중 하나로 분류합니다
              </p>

              {/* 업로드 영역: 드래그 앤 드롭, 클릭, 붙여넣기 지원 */}
              <div
                className={`upload-zone ${isDragging ? 'dragging' : ''}`}
                onClick={() => fileInputRef.current?.click()}
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
              >
                {/* 숨겨진 파일 입력 요소 (프로그래매틱 클릭용) */}
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={(e) => handleFileSelect(e.target.files[0])}
                  style={{ display: 'none' }}
                />
                <div className="upload-icon">[ + ]</div>
                <p className="upload-text">
                  이미지를 드래그하거나 클릭하여 업로드
                </p>
                <p className="upload-hint">
                  또는 Ctrl+V로 붙여넣기
                </p>
              </div>
            </div>
          )}

          {/* 이미지 미리보기 및 분석 화면 */}
          {preview && (
            <div className="preview-section">
              {/* 업로드된 이미지 미리보기 */}
              <div className="image-container">
                <img src={preview} alt="Preview" className="preview-image" />
              </div>

              {/* 모드 선택: 음식 분류 vs 객체 탐지 */}
              {!result && (
                <div className="mode-selection">
                  <button
                    className={`mode-btn ${mode === 'classify' ? 'active' : ''}`}
                    onClick={() => setMode('classify')}
                  >
                    음식 분류
                  </button>
                  <button
                    className={`mode-btn ${mode === 'detect' ? 'active' : ''}`}
                    onClick={() => setMode('detect')}
                  >
                    객체 탐지 (YOLO)
                  </button>
                </div>
              )}

              {/* 분석 전 컨트롤 버튼들 */}
              {!result && (
                <>
                  {/* Grad-CAM 옵션 (음식 분류 모드에서만 표시) */}
                  {mode === 'classify' && (
                    <div className="gradcam-toggle">
                      <label className="checkbox-label">
                        <input
                          type="checkbox"
                          checked={showGradCAM}
                          onChange={(e) => setShowGradCAM(e.target.checked)}
                        />
                        <span>판단 근거 히트맵 보기 (Grad-CAM)</span>
                      </label>
                    </div>
                  )}
                  {/* 분석 및 초기화 버튼 */}
                  <div className="button-group">
                    <button
                      className="btn btn-primary"
                      onClick={handlePredict}
                      disabled={loading}
                    >
                      {loading ? (
                        <>
                          <span className="spinner"></span>
                          분석 중...
                        </>
                      ) : (
                        '분석하기'
                      )}
                    </button>
                    <button
                      className="btn btn-secondary"
                      onClick={handleReset}
                      disabled={loading}
                    >
                      다시 선택
                    </button>
                  </div>
                </>
              )}

              {/* 에러 메시지 표시 */}
              {error && (
                <div className="error-message">
                  <span className="error-icon">[!]</span>
                  {error}
                </div>
              )}

              {/* 분석 결과 표시 영역 */}
              {result && (
                <div className="results-section">
                  {/* 결과 헤더: 제목과 초기화 버튼 */}
                  <div className="result-header">
                    <h3>{mode === 'detect' ? '객체 탐지 결과' : '분류 결과'}</h3>
                    <button className="btn-reset" onClick={handleReset}>
                      새로운 이미지 분석
                    </button>
                  </div>

                  {/* YOLO 객체 탐지 결과 (detect 모드일 때만 표시) */}
                  {mode === 'detect' && result.detections && (
                    <>
                      {/* 바운딩 박스가 그려진 어노테이션 이미지 */}
                      <div className="detection-section">
                        <h4 className="detection-title">
                          탐지된 객체 ({result.num_objects}개)
                        </h4>
                        <div className="detection-image-container">
                          <img
                            src={result.annotated_image}
                            alt="YOLO Detection Result"
                            className="detection-image"
                          />
                        </div>
                      </div>

                      {/* 탐지된 객체 상세 목록 */}
                      {result.num_objects > 0 && (
                        <div className="detection-list">
                          <h4>탐지된 객체 목록</h4>
                          {result.detections.map((detection, idx) => (
                            <div key={idx} className="detection-item">
                              {/* 객체 정보: 순위, 클래스, 신뢰도 */}
                              <div className="detection-info">
                                <span className="detection-rank">#{idx + 1}</span>
                                <span className="detection-class">{detection.class}</span>
                                <span className="detection-confidence">
                                  {detection.confidence_percent}
                                </span>
                              </div>
                              {/* 바운딩 박스 위치 및 크기 정보 */}
                              <div className="detection-bbox">
                                <small>
                                  위치: ({detection.bbox.x1.toFixed(0)}, {detection.bbox.y1.toFixed(0)}) -
                                  크기: {detection.bbox.width.toFixed(0)} × {detection.bbox.height.toFixed(0)}
                                </small>
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </>
                  )}

                  {/* Grad-CAM 히트맵 (showGradCAM이 활성화되었을 때만 표시) */}
                  {result.gradcam && (
                    <div className="gradcam-section">
                      <h4 className="gradcam-title">AI 판단 근거 히트맵</h4>
                      <p className="gradcam-description">
                        {result.gradcam.description}
                      </p>
                      {/* 원본 이미지에 히트맵이 오버레이된 이미지 */}
                      <div className="gradcam-container">
                        <img
                          src={result.gradcam.heatmap_image}
                          alt="Grad-CAM Heatmap"
                          className="gradcam-image"
                        />
                      </div>
                    </div>
                  )}

                  {/* 음식 분류 결과 (classify 모드일 때만 표시) */}
                  {result.prediction && (
                    <>
                      {/* 메인 예측 결과: Top-1 예측 */}
                      <div className="main-result">
                        <div className="result-label">예측된 음식</div>
                        <div className="result-value">
                          {/* 언더스코어를 공백으로 변환 (apple_pie -> apple pie) */}
                          {result.prediction.class.replace(/_/g, ' ')}
                        </div>
                        <div className="result-confidence">
                          확신도: {result.prediction.confidence_percent}
                        </div>
                      </div>

                      {/* Top-5 예측 결과 목록 */}
                      {result.top5 && (
                        <div className="top5-section">
                          <h4 className="top5-title">상위 5개 예측</h4>
                          <div className="top5-list">
                            {result.top5.map((item) => (
                              <div key={item.rank} className="top5-item">
                                {/* 순위 표시 */}
                                <span className="rank">#{item.rank}</span>
                                {/* 클래스 이름 (언더스코어 제거) */}
                                <span className="class-name">
                                  {item.class.replace(/_/g, ' ')}
                                </span>
                                {/* 신뢰도 프로그레스 바 */}
                                <div className="progress-bar">
                                  <div
                                    className="progress-fill"
                                    style={{ width: `${item.confidence * 100}%` }}
                                  ></div>
                                </div>
                                {/* 신뢰도 퍼센트 텍스트 */}
                                <span className="percentage">
                                  {item.confidence_percent}
                                </span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </main>

      {/* 푸터: 모델 정보 표시 */}
      <footer className="footer">
        <p>Powered by ResNet18 - 76.32% Accuracy - 101 Food Classes</p>
      </footer>
    </div>
  );
}

export default App;
