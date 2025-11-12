/**
 * 음식 분류 결과 표시 컴포넌트
 */

import React from 'react';

const ClassificationResult = ({ prediction, top5, gradcam }) => {
  return (
    <>
      {/* Grad-CAM 히트맵 */}
      {gradcam && (
        <div className="gradcam-section">
          <h4 className="gradcam-title">AI 판단 근거 히트맵</h4>
          <p className="gradcam-description">
            {gradcam.description}
          </p>
          <div className="gradcam-container">
            <img
              src={gradcam.heatmap_image}
              alt="Grad-CAM Heatmap"
              className="gradcam-image"
            />
          </div>
        </div>
      )}

      {/* 메인 예측 결과 */}
      <div className="main-result">
        <div className="result-label">예측된 음식</div>
        <div className="result-value">
          {prediction.class.replace(/_/g, ' ')}
        </div>
        <div className="result-confidence">
          확신도: {prediction.confidence_percent}
        </div>
      </div>

      {/* Top-5 예측 결과 */}
      {top5 && (
        <div className="top5-section">
          <h4 className="top5-title">상위 5개 예측</h4>
          <div className="top5-list">
            {top5.map((item) => (
              <div key={item.rank} className="top5-item">
                <span className="rank">#{item.rank}</span>
                <span className="class-name">
                  {item.class.replace(/_/g, ' ')}
                </span>
                <div className="progress-bar">
                  <div
                    className="progress-fill"
                    style={{ width: `${item.confidence * 100}%` }}
                  ></div>
                </div>
                <span className="percentage">
                  {item.confidence_percent}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </>
  );
};

export default ClassificationResult;
