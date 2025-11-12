#!/bin/bash

# Food-101 프로젝트 통합 실행 스크립트
# 사용법: ./run.sh [setup|start|stop|restart|status|train]

set -e  # 에러 발생 시 즉시 종료

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 프로젝트 루트 디렉토리
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$PROJECT_ROOT/api"
FRONTEND_DIR="$PROJECT_ROOT/frontend"
MODEL_PATH="$PROJECT_ROOT/outputs/models/best_model.pth"

# PID 파일
BACKEND_PID_FILE="$PROJECT_ROOT/.backend.pid"
FRONTEND_PID_FILE="$PROJECT_ROOT/.frontend.pid"

# 로그 함수
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 배너 출력
print_banner() {
    echo -e "${GREEN}"
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║                                                           ║"
    echo "║           Food-101 Image Classification                   ║"
    echo "║              프로젝트 관리 스크립트                        ║"
    echo "║                                                           ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# 사용법 출력
print_usage() {
    echo "사용법: ./run.sh [명령어]"
    echo ""
    echo "명령어:"
    echo "  setup      - 초기 환경 설정 (의존성 설치)"
    echo "  start      - 백엔드 + 프론트엔드 실행"
    echo "  stop       - 실행 중인 서버 종료"
    echo "  restart    - 서버 재시작"
    echo "  status     - 서버 상태 확인"
    echo "  train      - 모델 훈련 시작"
    echo "  logs       - 로그 확인"
    echo "  clean      - PID 파일 및 임시 파일 정리"
    echo ""
    echo "예시:"
    echo "  ./run.sh setup    # 처음 실행 시"
    echo "  ./run.sh start    # 서버 시작"
    echo "  ./run.sh status   # 상태 확인"
}

# 환경 확인
check_environment() {
    log_info "환경 확인 중..."

    # Python 확인
    if ! command -v python3 &> /dev/null; then
        log_error "Python3가 설치되어 있지 않습니다."
        exit 1
    fi
    log_success "Python3: $(python3 --version)"

    # Node.js 확인
    if ! command -v node &> /dev/null; then
        log_error "Node.js가 설치되어 있지 않습니다."
        exit 1
    fi
    log_success "Node.js: $(node --version)"

    # npm 확인
    if ! command -v npm &> /dev/null; then
        log_error "npm이 설치되어 있지 않습니다."
        exit 1
    fi
    log_success "npm: $(npm --version)"

    # GPU 확인 (선택사항)
    if command -v nvidia-smi &> /dev/null; then
        log_success "GPU: 사용 가능"
        nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1
    else
        log_warning "GPU: 사용 불가 (CPU 모드로 실행됩니다)"
    fi

    # 모델 파일 확인
    if [ -f "$MODEL_PATH" ]; then
        log_success "모델 파일: 존재함 ($MODEL_PATH)"
    else
        log_warning "모델 파일: 없음 (훈련이 필요합니다)"
        log_warning "실행 명령: ./run.sh train"
    fi
}

# 초기 설정
setup() {
    log_info "초기 환경 설정을 시작합니다..."

    check_environment

    # 백엔드 의존성 설치
    log_info "백엔드 의존성 설치 중..."
    cd "$BACKEND_DIR"
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        log_success "백엔드 의존성 설치 완료"
    else
        log_error "requirements.txt를 찾을 수 없습니다."
        exit 1
    fi

    # 프론트엔드 의존성 설치
    log_info "프론트엔드 의존성 설치 중..."
    cd "$FRONTEND_DIR"
    if [ -f "package.json" ]; then
        npm install
        log_success "프론트엔드 의존성 설치 완료"
    else
        log_error "package.json을 찾을 수 없습니다."
        exit 1
    fi

    cd "$PROJECT_ROOT"
    log_success "초기 설정이 완료되었습니다!"
    echo ""
    log_info "서버를 시작하려면: ./run.sh start"
}

# 백엔드 시작
start_backend() {
    log_info "백엔드 서버 시작 중..."

    # 모델 파일 확인
    if [ ! -f "$MODEL_PATH" ]; then
        log_error "모델 파일이 없습니다: $MODEL_PATH"
        log_error "먼저 모델을 훈련하세요: ./run.sh train"
        exit 1
    fi

    cd "$BACKEND_DIR"
    nohup python main.py > "$PROJECT_ROOT/backend.log" 2>&1 &
    BACKEND_PID=$!
    echo $BACKEND_PID > "$BACKEND_PID_FILE"

    # 서버가 시작될 때까지 대기
    sleep 3

    if ps -p $BACKEND_PID > /dev/null; then
        log_success "백엔드 서버가 시작되었습니다 (PID: $BACKEND_PID)"
        log_info "백엔드 URL: http://localhost:8000"
        log_info "API 문서: http://localhost:8000/docs"
    else
        log_error "백엔드 서버 시작 실패"
        log_error "로그 확인: cat $PROJECT_ROOT/backend.log"
        exit 1
    fi
}

# 프론트엔드 시작
start_frontend() {
    log_info "프론트엔드 서버 시작 중..."

    cd "$FRONTEND_DIR"
    nohup npm start > "$PROJECT_ROOT/frontend.log" 2>&1 &
    FRONTEND_PID=$!
    echo $FRONTEND_PID > "$FRONTEND_PID_FILE"

    # 서버가 시작될 때까지 대기
    log_info "프론트엔드가 빌드되고 있습니다 (약 10-20초 소요)..."
    sleep 15

    if ps -p $FRONTEND_PID > /dev/null; then
        log_success "프론트엔드 서버가 시작되었습니다 (PID: $FRONTEND_PID)"
        log_info "프론트엔드 URL: http://localhost:3000"
    else
        log_error "프론트엔드 서버 시작 실패"
        log_error "로그 확인: cat $PROJECT_ROOT/frontend.log"
        exit 1
    fi
}

# 서버 시작
start() {
    log_info "서버를 시작합니다..."

    # 이미 실행 중인지 확인
    if [ -f "$BACKEND_PID_FILE" ] && ps -p $(cat "$BACKEND_PID_FILE") > /dev/null 2>&1; then
        log_warning "백엔드가 이미 실행 중입니다."
    else
        start_backend
    fi

    if [ -f "$FRONTEND_PID_FILE" ] && ps -p $(cat "$FRONTEND_PID_FILE") > /dev/null 2>&1; then
        log_warning "프론트엔드가 이미 실행 중입니다."
    else
        start_frontend
    fi

    echo ""
    log_success "모든 서버가 시작되었습니다!"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${GREEN}🚀 애플리케이션이 실행 중입니다!${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${BLUE}📱 프론트엔드:${NC} http://localhost:3000"
    echo -e "${BLUE}🔧 백엔드:${NC}     http://localhost:8000"
    echo -e "${BLUE}📚 API 문서:${NC}   http://localhost:8000/docs"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "서버를 종료하려면: ./run.sh stop"
    echo "상태를 확인하려면: ./run.sh status"
    echo "로그를 보려면: ./run.sh logs"
}

# 서버 종료
stop() {
    log_info "서버를 종료합니다..."

    # 백엔드 종료
    if [ -f "$BACKEND_PID_FILE" ]; then
        BACKEND_PID=$(cat "$BACKEND_PID_FILE")
        if ps -p $BACKEND_PID > /dev/null 2>&1; then
            kill $BACKEND_PID
            log_success "백엔드 서버 종료됨 (PID: $BACKEND_PID)"
        else
            log_warning "백엔드 서버가 실행 중이 아닙니다."
        fi
        rm -f "$BACKEND_PID_FILE"
    else
        log_warning "백엔드 PID 파일을 찾을 수 없습니다."
    fi

    # 프론트엔드 종료
    if [ -f "$FRONTEND_PID_FILE" ]; then
        FRONTEND_PID=$(cat "$FRONTEND_PID_FILE")
        if ps -p $FRONTEND_PID > /dev/null 2>&1; then
            kill $FRONTEND_PID
            log_success "프론트엔드 서버 종료됨 (PID: $FRONTEND_PID)"
        else
            log_warning "프론트엔드 서버가 실행 중이 아닙니다."
        fi
        rm -f "$FRONTEND_PID_FILE"
    else
        log_warning "프론트엔드 PID 파일을 찾을 수 없습니다."
    fi

    # Node 프로세스 강제 종료 (필요한 경우)
    pkill -f "react-scripts start" 2>/dev/null || true

    log_success "모든 서버가 종료되었습니다."
}

# 서버 재시작
restart() {
    log_info "서버를 재시작합니다..."
    stop
    sleep 2
    start
}

# 상태 확인
status() {
    log_info "서버 상태를 확인합니다..."
    echo ""

    # 백엔드 상태
    if [ -f "$BACKEND_PID_FILE" ]; then
        BACKEND_PID=$(cat "$BACKEND_PID_FILE")
        if ps -p $BACKEND_PID > /dev/null 2>&1; then
            echo -e "${GREEN}✓${NC} 백엔드: 실행 중 (PID: $BACKEND_PID)"
            echo "  URL: http://localhost:8000"

            # 백엔드 헬스 체크
            if command -v curl &> /dev/null; then
                if curl -s http://localhost:8000/health > /dev/null 2>&1; then
                    echo -e "  ${GREEN}헬스 체크: 정상${NC}"
                else
                    echo -e "  ${YELLOW}헬스 체크: 응답 없음${NC}"
                fi
            fi
        else
            echo -e "${RED}✗${NC} 백엔드: 중지됨"
        fi
    else
        echo -e "${RED}✗${NC} 백엔드: 실행 중이 아님"
    fi

    echo ""

    # 프론트엔드 상태
    if [ -f "$FRONTEND_PID_FILE" ]; then
        FRONTEND_PID=$(cat "$FRONTEND_PID_FILE")
        if ps -p $FRONTEND_PID > /dev/null 2>&1; then
            echo -e "${GREEN}✓${NC} 프론트엔드: 실행 중 (PID: $FRONTEND_PID)"
            echo "  URL: http://localhost:3000"
        else
            echo -e "${RED}✗${NC} 프론트엔드: 중지됨"
        fi
    else
        echo -e "${RED}✗${NC} 프론트엔드: 실행 중이 아님"
    fi

    echo ""

    # 모델 상태
    if [ -f "$MODEL_PATH" ]; then
        MODEL_SIZE=$(du -h "$MODEL_PATH" | cut -f1)
        echo -e "${GREEN}✓${NC} 모델 파일: 존재함 ($MODEL_SIZE)"
    else
        echo -e "${YELLOW}⚠${NC} 모델 파일: 없음 (훈련 필요)"
    fi

    echo ""

    # GPU 상태
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU 상태:"
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
        awk -F, '{printf "  GPU %s: %s (%s%% 사용, 메모리: %s/%s MB)\n", $1, $2, $3, $4, $5}'
    fi
}

# 로그 확인
logs() {
    echo "로그를 확인합니다..."
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "어떤 로그를 보시겠습니까?"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "1) 백엔드 로그"
    echo "2) 프론트엔드 로그"
    echo "3) 모든 로그"
    echo "4) 실시간 로그 (tail -f)"
    echo ""
    read -p "선택 (1-4): " choice

    case $choice in
        1)
            if [ -f "$PROJECT_ROOT/backend.log" ]; then
                tail -n 50 "$PROJECT_ROOT/backend.log"
            else
                log_error "백엔드 로그 파일이 없습니다."
            fi
            ;;
        2)
            if [ -f "$PROJECT_ROOT/frontend.log" ]; then
                tail -n 50 "$PROJECT_ROOT/frontend.log"
            else
                log_error "프론트엔드 로그 파일이 없습니다."
            fi
            ;;
        3)
            echo "=== 백엔드 로그 ==="
            [ -f "$PROJECT_ROOT/backend.log" ] && tail -n 25 "$PROJECT_ROOT/backend.log"
            echo ""
            echo "=== 프론트엔드 로그 ==="
            [ -f "$PROJECT_ROOT/frontend.log" ] && tail -n 25 "$PROJECT_ROOT/frontend.log"
            ;;
        4)
            log_info "실시간 로그 (Ctrl+C로 종료)"
            tail -f "$PROJECT_ROOT/backend.log" "$PROJECT_ROOT/frontend.log" 2>/dev/null
            ;;
        *)
            log_error "잘못된 선택입니다."
            ;;
    esac
}

# 모델 훈련
train() {
    log_info "모델 훈련을 시작합니다..."

    if [ -f "$PROJECT_ROOT/notebooks/food101_training.py" ]; then
        cd "$PROJECT_ROOT/notebooks"
        python food101_training.py
    elif [ -f "$PROJECT_ROOT/notebooks/food101_training.ipynb" ]; then
        log_info "Jupyter Notebook을 실행합니다..."
        cd "$PROJECT_ROOT/notebooks"
        jupyter notebook food101_training.ipynb
    else
        log_error "훈련 스크립트를 찾을 수 없습니다."
        exit 1
    fi
}

# 정리
clean() {
    log_info "임시 파일을 정리합니다..."

    rm -f "$BACKEND_PID_FILE"
    rm -f "$FRONTEND_PID_FILE"
    rm -f "$PROJECT_ROOT/backend.log"
    rm -f "$PROJECT_ROOT/frontend.log"

    log_success "정리 완료"
}

# 메인 로직
main() {
    print_banner

    # 인자가 없으면 사용법 출력
    if [ $# -eq 0 ]; then
        print_usage
        exit 0
    fi

    # 명령어 처리
    case "$1" in
        setup)
            setup
            ;;
        start)
            start
            ;;
        stop)
            stop
            ;;
        restart)
            restart
            ;;
        status)
            status
            ;;
        train)
            train
            ;;
        logs)
            logs
            ;;
        clean)
            clean
            ;;
        *)
            log_error "알 수 없는 명령어: $1"
            echo ""
            print_usage
            exit 1
            ;;
    esac
}

# 스크립트 실행
main "$@"
