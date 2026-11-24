# 로컬 커밋과 이후 공개 절차

제출 파일을 검토한 뒤 사용자 요청에 따라 로컬 Git에 커밋하였다. 이후 사용자가 만든 다음 공개 저장소를 업로드 대상으로 지정하였다.

https://github.com/firesmr11/2022041038-PARK-JEONGHWAN-machine-learning-YOLO-fine-tuning
작업 폴더는 이 저장소의 루트이며, 가상환경·임시 파일·기존 output ZIP·계정 키는 포함 대상에서 제외하였다.

현재 커밋과 작업 상태를 확인한다.

```powershell
git status
git log -1 --oneline
git show --stat HEAD
```

실제 공개 주소는 README.md와 docs/report_1_3_to_9.txt에 반영하였다.
기존 HWP에 보고서를 옮긴 뒤 표지·1.1·1.2와 합쳐 최종 보고서를 저장한다.
강의자료의 정확한 5초 문구와 현재 6.72초 전체 2배속 영상의 차이는 docs/submission_checklist.md에 남겨 두었다.

공개 주소 등을 수정한 뒤 추가 커밋할 때의 예시는 다음과 같다.

```powershell
git add README.md docs/report_1_3_to_9.txt
git commit -m "Add public repository link"
```

지정한 원격 주소와 GitHub의 main 브랜치를 사용한다. 기존 파일을 덮어쓰는 강제 push는 사용하지 않는다.

```powershell
git remote add origin https://github.com/firesmr11/2022041038-PARK-JEONGHWAN-machine-learning-YOLO-fine-tuning.git
git push -u origin HEAD:main
```

이미 origin이 있는 다른 컴퓨터에서는 먼저 git remote -v로 확인하고 중복 추가하지 않는다.
업로드 후 로그아웃 상태에서도 모델·영상·README를 볼 수 있는지 확인하고 보고서에 공개 주소를 기재한다.
