/**
 * API 서비스 레이어
 * 모든 백엔드 API 호출을 중앙화하여 관리
 */

import axios from 'axios';

// API 기본 URL
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Axios 인스턴스 생성 (공통 설정)
const apiClient = axios.create({
  baseURL: API_URL,
  timeout: 30000, // 30초 타임아웃
  headers: {
    'Content-Type': 'multipart/form-data'
  }
});

// 요청 인터셉터 (로깅)
apiClient.interceptors.request.use(
  (config) => {
    if (process.env.REACT_APP_DEBUG === 'true') {
      console.log('[API Request]', config.method.toUpperCase(), config.url);
    }
    return config;
  },
  (error) => {
    console.error('[API Request Error]', error);
    return Promise.reject(error);
  }
);

// 응답 인터셉터 (에러 처리)
apiClient.interceptors.response.use(
  (response) => {
    if (process.env.REACT_APP_DEBUG === 'true') {
      console.log('[API Response]', response.status, response.data);
    }
    return response;
  },
  (error) => {
    console.error('[API Response Error]', error.response?.data || error.message);

    // 사용자 친화적인 에러 메시지 생성
    let errorMessage = '요청 처리 중 오류가 발생했습니다.';

    if (error.response) {
      // 서버에서 응답을 받았지만 에러 상태 코드
      const status = error.response.status;
      const detail = error.response.data?.detail;

      if (status === 413) {
        errorMessage = '파일 크기가 너무 큽니다. 10MB 이하의 파일을 업로드해주세요.';
      } else if (status === 415) {
        errorMessage = '지원하지 않는 파일 형식입니다. JPEG, PNG 파일을 사용해주세요.';
      } else if (status === 503) {
        errorMessage = '서버가 준비되지 않았습니다. 잠시 후 다시 시도해주세요.';
      } else if (detail) {
        errorMessage = detail;
      }
    } else if (error.request) {
      // 요청은 보냈지만 응답을 받지 못함
      errorMessage = '서버에 연결할 수 없습니다. 네트워크 연결을 확인해주세요.';
    }

    error.userMessage = errorMessage;
    return Promise.reject(error);
  }
);


/**
 * API 서비스 객체
 */
const api = {
  /**
   * 서버 헬스 체크
   * @returns {Promise<Object>} 서버 상태 정보
   */
  healthCheck: async () => {
    const response = await apiClient.get('/health');
    return response.data;
  },

  /**
   * 클래스 목록 조회
   * @returns {Promise<Object>} 101개 음식 클래스 목록
   */
  getClasses: async () => {
    const response = await apiClient.get('/classes');
    return response.data;
  },

  /**
   * 음식 이미지 분류 (기본)
   * @param {File} file - 업로드할 이미지 파일
   * @returns {Promise<Object>} 예측 결과
   */
  predictFood: async (file) => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await apiClient.post('/predict', formData);
    return response.data;
  },

  /**
   * 음식 이미지 분류 + Grad-CAM
   * @param {File} file - 업로드할 이미지 파일
   * @returns {Promise<Object>} 예측 결과 + 히트맵
   */
  predictFoodWithGradCAM: async (file) => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await apiClient.post('/predict/gradcam', formData);
    return response.data;
  },

  /**
   * YOLO 객체 탐지
   * @param {File} file - 업로드할 이미지 파일
   * @param {number} confThreshold - 신뢰도 임계값 (기본값: 0.25)
   * @returns {Promise<Object>} 탐지 결과
   */
  detectObjects: async (file, confThreshold = 0.25) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('conf_threshold', confThreshold);

    const response = await apiClient.post('/detect', formData);
    return response.data;
  },

  /**
   * YOLO 클래스 목록 조회
   * @returns {Promise<Object>} 80개 COCO 클래스 목록
   */
  getYOLOClasses: async () => {
    const response = await apiClient.get('/detect/classes');
    return response.data;
  }
};

export default api;
