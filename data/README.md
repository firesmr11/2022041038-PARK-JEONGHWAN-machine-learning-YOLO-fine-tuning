# 데이터 구성 및 준비

직접 촬영한 컵(`cup`), 마우스(`mouse`), 리모컨(`remote`)을 Roboflow에서 라벨링하였다.
클래스 ID는 0=cup, 1=mouse, 2=remote이다. 전체 사진과 라벨 ZIP은 이 저장소에 포함하지 않았다.
원본 데이터의 Roboflow 라이선스 표기는 Private이며, 공개된 다운로드 주소나 API 키는 제공하지 않는다.
데이터 접근 권한이 있는 사용자는 해당 프로젝트의 버전 1과 2를 **YOLOv8 / Download zip to computer**로 내려받는다.

| 버전 | 학습 이미지 | 검증 이미지 | 테스트 이미지 | 전체 |
|---|---:|---:|---:|---:|
| v1 | 196 | 39 | 19 | 254 |
| v2 | 414 | 39 | 19 | 472 |

| 버전·분할 | cup 박스 | mouse 박스 | remote 박스 |
|---|---:|---:|---:|
| v1 train | 74 | 103 | 75 |
| v2 train | 135 | 212 | 123 |
| 공통 val | 17 | 13 | 20 |
| 공통 test | 10 | 6 | 10 |

위 클래스별 수치는 **객체 박스 수**이며 클래스별 이미지 수가 아니다.
v1의 196장 학습셋은 v2의 학습셋에 포함되어 있고 검증·테스트의 이미지와 라벨은 유지하였다.
원래 로컬 실험 기록에 해시 비교 통과가 남아 있다. 현재 저장소에는 원본 데이터와 전체 해시 목록이 없어
그 주장을 독립 재검증한 자료를 포함하지 않는다. 아래 코드로 보유한 두 ZIP을 다시 검사할 수 있다.

압축 해제 예시:

```text
datasets/v1/data.yaml
datasets/v1/train/images/  train/labels/
datasets/v1/valid/images/  valid/labels/
datasets/v1/test/images/   test/labels/
datasets/v2/...            같은 구조
```

```powershell
python scripts/audit_dataset.py --data datasets/v1/data.yaml --compare datasets/v2/data.yaml
```

이미지/라벨 해시가 같아도 유사 장면이나 연속 촬영 사진의 분할 간 중복까지 배제되는 것은 아니다.
추가 수집 시에는 기존 검증·테스트를 학습셋에 섞지 않고, 촬영 세션 단위로 분리하는 것이 바람직하다.
