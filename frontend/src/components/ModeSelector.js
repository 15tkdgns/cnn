/**
 * 모드 선택 컴포넌트
 * 음식 분류 vs 객체 탐지 선택
 */

import React from 'react';

const ModeSelector = ({ mode, onModeChange }) => {
  return (
    <div className="mode-selection">
      <button
        className={`mode-btn ${mode === 'classify' ? 'active' : ''}`}
        onClick={() => onModeChange('classify')}
      >
        음식 분류
      </button>
      <button
        className={`mode-btn ${mode === 'detect' ? 'active' : ''}`}
        onClick={() => onModeChange('detect')}
      >
        객체 탐지 (YOLO)
      </button>
    </div>
  );
};

export default ModeSelector;
