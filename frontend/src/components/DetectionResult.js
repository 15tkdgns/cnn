/**
 * YOLO 객체 탐지 결과 표시 컴포넌트
 */

import React from 'react';

const DetectionResult = ({ numObjects, detections, annotatedImage }) => {
  return (
    <>
      {/* 어노테이션된 이미지 */}
      <div className="detection-section">
        <h4 className="detection-title">
          탐지된 객체 ({numObjects}개)
        </h4>
        <div className="detection-image-container">
          <img
            src={annotatedImage}
            alt="YOLO Detection Result"
            className="detection-image"
          />
        </div>
      </div>

      {/* 탐지된 객체 목록 */}
      {numObjects > 0 && (
        <div className="detection-list">
          <h4>탐지된 객체 목록</h4>
          {detections.map((detection, idx) => (
            <div key={idx} className="detection-item">
              <div className="detection-info">
                <span className="detection-rank">#{idx + 1}</span>
                <span className="detection-class">{detection.class}</span>
                <span className="detection-confidence">
                  {detection.confidence_percent}
                </span>
              </div>
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
  );
};

export default DetectionResult;
