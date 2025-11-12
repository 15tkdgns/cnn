/**
 * 이미지 업로드 커스텀 훅
 * 파일 선택, 드래그 앤 드롭, 붙여넣기 로직을 재사용 가능하게 캡슐화
 */

import { useState, useCallback } from 'react';

const MAX_FILE_SIZE = (process.env.REACT_APP_MAX_FILE_SIZE_MB || 10) * 1024 * 1024;

const useImageUpload = () => {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState(null);

  /**
   * 파일 유효성 검증
   */
  const validateFile = useCallback((file) => {
    if (!file) {
      setError('파일이 선택되지 않았습니다.');
      return false;
    }

    // 파일 타입 검증
    if (!file.type.startsWith('image/')) {
      setError('이미지 파일만 업로드 가능합니다.');
      return false;
    }

    // 파일 크기 검증
    if (file.size > MAX_FILE_SIZE) {
      const maxMB = MAX_FILE_SIZE / (1024 * 1024);
      setError(`파일 크기는 ${maxMB}MB 이하여야 합니다.`);
      return false;
    }

    return true;
  }, []);

  /**
   * 파일 선택 처리
   */
  const handleFileSelect = useCallback((file) => {
    setError(null);

    if (!validateFile(file)) {
      return;
    }

    setImage(file);

    // FileReader로 미리보기 생성
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target.result);
    reader.onerror = () => setError('파일을 읽을 수 없습니다.');
    reader.readAsDataURL(file);
  }, [validateFile]);

  /**
   * 드래그 앤 드롭 핸들러
   */
  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);

    const file = e.dataTransfer.files[0];
    handleFileSelect(file);
  }, [handleFileSelect]);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setIsDragging(false);
  }, []);

  /**
   * 붙여넣기 핸들러
   */
  const handlePaste = useCallback((e) => {
    const items = e.clipboardData.items;
    for (let item of items) {
      if (item.type.indexOf('image') !== -1) {
        const file = item.getAsFile();
        handleFileSelect(file);
        break;
      }
    }
  }, [handleFileSelect]);

  /**
   * 초기화
   */
  const reset = useCallback(() => {
    setImage(null);
    setPreview(null);
    setError(null);
    setIsDragging(false);
  }, []);

  return {
    image,
    preview,
    isDragging,
    error,
    handleFileSelect,
    handleDrop,
    handleDragOver,
    handleDragLeave,
    handlePaste,
    reset,
    setError
  };
};

export default useImageUpload;
