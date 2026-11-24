# 제공된 학습 모델

파일 이름은 학습 환경·학습 이미지 수·요청 에포크 수를 뜻한다. 모두 검증 기준으로 선택한 best.pt이다.

| 모델 | 테스트 mAP50-95 | 용도 |
|---|---:|---|
| local_196_25_best.pt | 95.81% | 로컬 비교 |
| local_196_50_best.pt | 97.97% | 로컬 비교 |
| local_196_75_best.pt | 98.73% | 최종 시연 후보 |
| local_414_25_best.pt | 97.63% | 로컬 비교 |
| local_414_50_best.pt | 97.80% | 로컬 비교 |
| local_414_75_best.pt | 96.94% | 로컬 비교 |
| colab_414_50_best.pt | 98.50% | Colab 최종 비교 후보 |

Colab의 나머지 5개 체크포인트는 현재 제공받은 자료에 없어 포함하지 않았다. 해당 실험의 지표는 results/test_comparison.csv에 있다.
파일 체크섬은 SHA256SUMS.txt, 저장된 검증 지표와 설정은 results/checkpoint_inventory.json에 기록하였다.
테스트 최고 결과를 참고해 최종 후보를 골랐으므로, 별도의 새 테스트셋에서 다시 평가할 필요가 있다.
