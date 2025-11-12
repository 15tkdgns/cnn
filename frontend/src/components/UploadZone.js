/**
 * 이미지 업로드 영역 컴포넌트
 * 드래그 앤 드롭, 클릭 업로드 지원
 */

import React from 'react';

const UploadZone = ({
  isDragging,
  onDrop,
  onDragOver,
  onDragLeave,
  onClick,
  fileInputRef,
  onFileSelect
}) => {
  return (
    <div
      className={`upload-zone ${isDragging ? 'dragging' : ''}`}
      onClick={onClick}
      onDrop={onDrop}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={(e) => onFileSelect(e.target.files[0])}
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
  );
};

export default UploadZone;
