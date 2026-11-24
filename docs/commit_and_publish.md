# 저장소 관리

공개 저장소: https://github.com/firesmr11/2022041038-PARK-JEONGHWAN-machine-learning-YOLO-fine-tuning

GitHub의 기본 브랜치는 main이다. 아래 명령은 저장소 루트에서 실행한다.

## 변경 내용 확인

```powershell
git status
git log -1 --oneline
git show --stat HEAD
```

## 문서 수정 및 반영

```powershell
git add README.md docs/report_1_3_to_9.txt
git commit -m "Update project documentation"
git push origin HEAD:main
```

## 다른 컴퓨터에서 시작

```powershell
git clone https://github.com/firesmr11/2022041038-PARK-JEONGHWAN-machine-learning-YOLO-fine-tuning.git
```

패키지 설치와 실행 명령은 [실행 방법](usage.md)을 참고한다.
가상환경, 임시 파일, 데이터 ZIP과 API 키는 커밋 대상에서 제외한다.
