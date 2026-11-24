# 결과를 읽는 방법

- `results/test_comparison.csv`: 동일 테스트 19장·26객체의 12개 모델 결과. 모든 지표 단위는 %.
- `results/local_training_times.csv`: 로컬 실험 CSV의 학습 시간. train 호출 벽시계와 results.csv 마지막 time을 별도 열로 기록.
- `results/colab_per_class.csv`: 평가 로그에 표시된 클래스별 mAP50-95. 로그 반올림 수치이다.
- Colab 전체 비교표는 평가 결과표의 소수점 둘째 자리 기록을 사용했다. 로컬은 원본 CSV의 정밀도를 유지했다.
- Colab 196장·50회 학습 완료 로그의 0.151시간은 약 543.6초이지만 반올림된 학습 완료 시간이다. 로컬 CSV 시간과 측정 범위를 완전히 일치시킨 자료가 아니므로 정확한 GPU 배속으로 비교하지 않는다.
- `results/history/*_checkpoint_history.csv`: best.pt 안의 train_results를 추출한 기록. 보관한 7개 모델은 각각 지정 에포크 수만큼 행이 있고 마지막 epoch도 완료 횟수와 일치했다. 원래 results.csv 파일 자체를 확보한 것으로 표기하지 않고 체크포인트에서 복원한 이력으로 구분하였다.
- `results/checkpoint_inventory.json`: 모델 해시, 저장된 검증 지표, 학습 설정 일부, 체크포인트에 포함된 이력 길이.
- `results/video/`: 새 13.43초 영상에 대한 프레임별 예측 박스·ID·신뢰도와 요약.
- 영상 출력 빈도는 정답 기준 Precision/Recall이 아니며, ID 수는 정식 ID 전환 오류 수가 아니다.
- 단일 seed, 작은 테스트셋, 같은 물체 및 유사 배경, 테스트를 참고한 최종 후보 선택이라는 한계가 있다.
- 학습·평가 스크립트는 기록된 실험 설정을 기준으로 재실행할 수 있도록 구성했다. OS, GPU 및 라이브러리 차이에 따라 같은 seed에서도 수치가 달라질 수 있다.
