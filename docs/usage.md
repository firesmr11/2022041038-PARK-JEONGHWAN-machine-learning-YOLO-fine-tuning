# 실행 방법

## 설치

Python 3.12 또는 3.13의 가상환경을 사용한다. 아래 명령은 저장소 루트의 PowerShell에서 실행한다.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
# 추론만 CPU로 실행할 때
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

RTX 4060에서 학습하려면 CPU 설치 명령 대신 다음과 같이 CUDA 빌드를 설치하고 실제 CUDA 사용 여부를 확인한다.
설치 환경에 맞는 빌드는 [PyTorch 공식 설치 안내](https://pytorch.org/get-started/previous-versions/)를 기준으로 선택한다.

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

이하 `python`은 위 가상환경의 Python을 의미한다. 활성화하지 않았다면 `python` 대신 `.\.venv\Scripts\python.exe`를 쓴다.
Colab은 [실행 노트북](../notebooks/colab_experiments.ipynb)을 이용한다.

## 데이터 준비와 학습

[데이터 준비 안내](../data/README.md)에 따라 Roboflow 버전 1·2를 YOLOv8 ZIP으로 별도 확보하고 `datasets/v1`, `datasets/v2`에 푼다.
원본 Roboflow 프로젝트는 Private이므로 저장소만 복제해 학습 데이터까지 자동으로 받을 수는 없다.
API 키와 데이터 ZIP은 커밋하지 않는다.

```powershell
python scripts/audit_dataset.py --data datasets/v1/data.yaml --compare datasets/v2/data.yaml
python scripts/train.py --data datasets/v1/data.yaml --epochs 25 --device 0 --name local_196_25
python scripts/train.py --data datasets/v1/data.yaml --epochs 50 --device 0 --name local_196_50
python scripts/train.py --data datasets/v1/data.yaml --epochs 75 --device 0 --name local_196_75
python scripts/train.py --data datasets/v2/data.yaml --epochs 25 --device 0 --name local_414_25
python scripts/train.py --data datasets/v2/data.yaml --epochs 50 --device 0 --name local_414_50
python scripts/train.py --data datasets/v2/data.yaml --epochs 75 --device 0 --name local_414_75
```

각 실행은 원래 `yolov8n.pt`에서 시작한다. 이전 실험의 best.pt를 이어 학습하지 않는다.
입력640, batch16, seed0, workers2, optimizer auto, patience0으로 실행하며 검증 성능으로 best.pt를 저장한다.
결과는 `runs/train/`에 저장되고 같은 이름을 재사용하면 별도 폴더가 생성된다.
첫 사전 학습 모델 다운로드에는 인터넷이 필요하다. GPU 학습 전 CUDA 사용 여부를 확인한다.

## 테스트셋 평가

```powershell
python scripts/evaluate.py --model models/local_196_75_best.pt --data datasets/v1/data.yaml --device cpu --name local_196_75
python scripts/evaluate.py --model models/colab_414_50_best.pt --data datasets/v2/data.yaml --device cpu --name colab_414_50
```

`split=test`를 명시해 학습 없이 평가한다. GPU가 있으면 `--device 0`을 사용할 수 있다.
결과 지표 JSON과 평가 그림은 `runs/evaluate/`에 저장된다.
학습 경로가 다른 컴퓨터의 절대 경로여도 현재 데이터 위치로 새 설정을 만들며 원본 data.yaml은 변경하지 않는다.

## 영상 추적 재실행

두 모델·원본 영상이 이미 포함되어 있어 데이터 ZIP 없이도 이 단계는 실행할 수 있다.

```powershell
python scripts/compare_video.py --video videos/original/desk_objects.mp4 --local models/local_196_75_best.pt --colab models/colab_414_50_best.pt --out runs/video_comparison --playback-speed 2
```

원속도와 전체 2배속 MP4, 비교 영상, CSV, JSON을 생성한다. 현재 비교 코드는 동일 CPU/FP32 조건으로 두 모델을 실행한다.
평가용 정답은 생성하지 않는다. `--inspect-only`를 추가하면 추론 없이 입력 정보와 표본 장면만 확인한다.

## 그래프 재생성과 기록 검증

```powershell
python scripts/plot_results.py
python scripts/verify_submission.py
python -m unittest discover -s tests -v
```

기존 테스트 수치를 새 평가 없이 덮어쓰지 않으며, 재실행 결과는 `runs/`에 분리한다.
모델 체크섬과 영상 길이·프레임 수, 결과표의 조건 조합을 검사할 수 있다.
