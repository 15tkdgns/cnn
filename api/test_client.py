"""
FastAPI 서버 테스트 클라이언트
API 엔드포인트를 테스트하기 위한 간단한 클라이언트
"""

import requests
from pathlib import Path
import json


def test_health_check(base_url="http://localhost:8000"):
    """헬스 체크 테스트"""
    print("\n" + "="*70)
    print("1. 헬스 체크 테스트")
    print("="*70)

    response = requests.get(f"{base_url}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")


def test_get_classes(base_url="http://localhost:8000"):
    """클래스 목록 가져오기"""
    print("\n" + "="*70)
    print("2. 클래스 목록 테스트")
    print("="*70)

    response = requests.get(f"{base_url}/classes")
    data = response.json()
    print(f"Status Code: {response.status_code}")
    print(f"총 클래스 수: {data['total']}")
    print(f"처음 10개 클래스: {data['classes'][:10]}")


def test_predict(image_path, base_url="http://localhost:8000"):
    """이미지 예측 테스트"""
    print("\n" + "="*70)
    print("3. 이미지 예측 테스트")
    print("="*70)
    print(f"이미지 파일: {image_path}")

    if not Path(image_path).exists():
        print(f"❌ 이미지 파일을 찾을 수 없습니다: {image_path}")
        return

    with open(image_path, "rb") as f:
        files = {"file": (Path(image_path).name, f, "image/jpeg")}
        response = requests.post(f"{base_url}/predict", files=files)

    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"\n✅ 예측 성공!")
        print(f"\n[Top-1 예측]")
        pred = data['prediction']
        print(f"  클래스: {pred['class']}")
        print(f"  확신도: {pred['confidence_percent']}")

        print(f"\n[Top-5 예측]")
        for item in data['top5']:
            print(f"  {item['rank']}. {item['class']}: {item['confidence_percent']}")
    else:
        print(f"❌ 예측 실패: {response.text}")


def test_batch_predict(image_paths, base_url="http://localhost:8000"):
    """배치 예측 테스트"""
    print("\n" + "="*70)
    print("4. 배치 예측 테스트")
    print("="*70)

    # 존재하는 파일만 필터링
    valid_paths = [p for p in image_paths if Path(p).exists()]

    if not valid_paths:
        print("❌ 유효한 이미지 파일이 없습니다")
        return

    print(f"이미지 파일 수: {len(valid_paths)}")

    files = [
        ("files", (Path(p).name, open(p, "rb"), "image/jpeg"))
        for p in valid_paths
    ]

    response = requests.post(f"{base_url}/predict/batch", files=files)

    # 파일 닫기
    for _, (_, f, _) in files:
        f.close()

    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"\n✅ 배치 예측 성공!")
        print(f"처리된 이미지: {data['total']}개\n")

        for result in data['results']:
            if 'error' in result:
                print(f"  ❌ {result['filename']}: {result['error']}")
            else:
                pred = result['prediction']
                print(f"  ✅ {result['filename']}: {pred['class']} ({pred['confidence_percent']})")
    else:
        print(f"❌ 배치 예측 실패: {response.text}")


def main():
    """메인 테스트 함수"""
    BASE_URL = "http://localhost:8000"

    print("="*70)
    print("Food-101 API 테스트 클라이언트")
    print("="*70)
    print(f"서버 URL: {BASE_URL}")

    try:
        # 1. 헬스 체크
        test_health_check(BASE_URL)

        # 2. 클래스 목록
        test_get_classes(BASE_URL)

        # 3. 단일 이미지 예측 (테스트 이미지 경로 지정)
        # 예시: Food-101 데이터셋의 이미지 사용
        test_image = Path(__file__).parent.parent / "data" / "food-101" / "food-101" / "images" / "apple_pie" / "1011328.jpg"

        # dataset_path.txt에서 경로 읽기
        dataset_path_file = Path(__file__).parent.parent / "data" / "dataset_path.txt"
        if dataset_path_file.exists():
            with open(dataset_path_file, 'r') as f:
                base_path = Path(f.read().strip())
            test_image = base_path / "food-101" / "food-101" / "images" / "apple_pie" / "1011328.jpg"

        if test_image.exists():
            test_predict(str(test_image), BASE_URL)
        else:
            print(f"\n⚠️  테스트 이미지를 찾을 수 없습니다: {test_image}")
            print("  사용자 이미지로 테스트하려면:")
            print(f"  test_predict('your_image.jpg', '{BASE_URL}')")

        # 4. 배치 예측 (선택사항)
        # test_batch_predict(['image1.jpg', 'image2.jpg'], BASE_URL)

    except requests.exceptions.ConnectionError:
        print("\n❌ 서버에 연결할 수 없습니다!")
        print(f"   서버가 실행 중인지 확인하세요: {BASE_URL}")
        print("\n서버 시작 방법:")
        print("  cd api")
        print("  python main.py")
        print("  또는")
        print("  uvicorn main:app --reload")

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")


if __name__ == "__main__":
    main()
