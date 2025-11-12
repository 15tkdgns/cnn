/**
 * 예측 API 호출 커스텀 훅
 * API 호출 상태 관리 및 재시도 로직 포함
 */

import { useState, useCallback } from 'react';
import api from '../services/api';

const usePrediction = () => {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  /**
   * 음식 분류 예측
   * @param {File} image - 이미지 파일
   * @param {boolean} withGradCAM - Grad-CAM 포함 여부
   */
  const predictFood = useCallback(async (image, withGradCAM = false) => {
    setLoading(true);
    setError(null);

    try {
      let response;
      if (withGradCAM) {
        response = await api.predictFoodWithGradCAM(image);
      } else {
        response = await api.predictFood(image);
      }

      setResult(response);
      return response;
    } catch (err) {
      const errorMessage = err.userMessage || '예측에 실패했습니다. 다시 시도해주세요.';
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  /**
   * YOLO 객체 탐지
   * @param {File} image - 이미지 파일
   * @param {number} confThreshold - 신뢰도 임계값
   */
  const detectObjects = useCallback(async (image, confThreshold = 0.25) => {
    setLoading(true);
    setError(null);

    try {
      const response = await api.detectObjects(image, confThreshold);
      setResult(response);
      return response;
    } catch (err) {
      const errorMessage = err.userMessage || '객체 탐지에 실패했습니다. 다시 시도해주세요.';
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  /**
   * 결과 초기화
   */
  const reset = useCallback(() => {
    setResult(null);
    setError(null);
  }, []);

  return {
    result,
    loading,
    error,
    predictFood,
    detectObjects,
    reset,
    setError
  };
};

export default usePrediction;
