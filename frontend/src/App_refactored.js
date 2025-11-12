/**
 * Food Classifier - React Frontend (리팩토링 버전)
 *
 * 개선 사항:
 * - 컴포넌트 분리로 가독성 향상
 * - 커스텀 훅으로 로직 재사용성 증가
 * - API 서비스 레이어 분리
 * - 타입 안전성 및 에러 처리 개선
 */

import React, { useState, useRef } from 'react';
import './App.css';

// 커스텀 훅
import useImageUpload from './hooks/useImageUpload';
import usePrediction from './hooks/usePrediction';

// 컴포넌트
import UploadZone from './components/UploadZone';
import ModeSelector from './components/ModeSelector';
import ClassificationResult from './components/ClassificationResult';
import DetectionResult from './components/DetectionResult';

function App() {
  // ===== State 관리 =====
  const [mode, setMode] = useState('classify');
  const [showGradCAM, setShowGradCAM] = useState(false);
  const fileInputRef = useRef(null);

  // 커스텀 훅 사용
  const {
    image,
    preview,
    isDragging,
    error: uploadError,
    handleFileSelect,
    handleDrop,
    handleDragOver,
    handleDragLeave,
    handlePaste,
    reset: resetUpload,
    setError: setUploadError
  } = useImageUpload();

  const {
    result,
    loading,
    error: predictionError,
    predictFood,
    detectObjects,
    reset: resetPrediction
  } = usePrediction();

  // 에러 통합 관리
  const error = uploadError || predictionError;

  // ===== 이벤트 핸들러 =====

  /**
   * 예측 수행
   */
  const handlePredict = async () => {
    if (!image) return;

    try {
      if (mode === 'detect') {
        await detectObjects(image);
      } else {
        await predictFood(image, showGradCAM);
      }
    } catch (err) {
      // 에러는 커스텀 훅에서 처리됨
      console.error('Prediction error:', err);
    }
  };

  /**
   * 초기화 처리
   */
  const handleReset = () => {
    resetUpload();
    resetPrediction();
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  /**
   * 모드 변경 처리
   */
  const handleModeChange = (newMode) => {
    setMode(newMode);
    resetPrediction();
  };

  // ===== JSX 렌더링 =====
  return (
    <div className="app" onPaste={handlePaste}>
      {/* 헤더 */}
      <header className="header">
        <div className="header-content">
          <h1 className="logo">
            {process.env.REACT_APP_NAME || 'Food Classifier'}
          </h1>
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

              <UploadZone
                isDragging={isDragging}
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onClick={() => fileInputRef.current?.click()}
                fileInputRef={fileInputRef}
                onFileSelect={handleFileSelect}
              />
            </div>
          )}

          {/* 이미지 미리보기 및 분석 화면 */}
          {preview && (
            <div className="preview-section">
              {/* 업로드된 이미지 미리보기 */}
              <div className="image-container">
                <img src={preview} alt="Preview" className="preview-image" />
              </div>

              {/* 모드 선택 (결과가 없을 때만 표시) */}
              {!result && (
                <ModeSelector mode={mode} onModeChange={handleModeChange} />
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
                  {/* 결과 헤더 */}
                  <div className="result-header">
                    <h3>{mode === 'detect' ? '객체 탐지 결과' : '분류 결과'}</h3>
                    <button className="btn-reset" onClick={handleReset}>
                      새로운 이미지 분석
                    </button>
                  </div>

                  {/* YOLO 객체 탐지 결과 */}
                  {mode === 'detect' && result.detections && (
                    <DetectionResult
                      numObjects={result.num_objects}
                      detections={result.detections}
                      annotatedImage={result.annotated_image}
                    />
                  )}

                  {/* 음식 분류 결과 */}
                  {result.prediction && (
                    <ClassificationResult
                      prediction={result.prediction}
                      top5={result.top5}
                      gradcam={result.gradcam}
                    />
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </main>

      {/* 푸터 */}
      <footer className="footer">
        <p>
          Powered by ResNet18 - 76.32% Accuracy - 101 Food Classes
          {process.env.REACT_APP_VERSION && ` | v${process.env.REACT_APP_VERSION}`}
        </p>
      </footer>
    </div>
  );
}

export default App;
