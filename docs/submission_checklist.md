# 실험 구성 및 결과 자료

| 항목 | 내용 | 위치 |
|---|---|---|
| 감지 대상 | cup·mouse·remote | [데이터 구성](../data/README.md) |
| 직접 촬영 데이터 | v1 254장 / v2 472장 | 데이터 ZIP은 별도 보관 |
| 학습 실험 | 196/414장 × 25/50/75회 × Colab/로컬, 총 12개 | [실험 설정](../configs/experiment.yaml) |
| 최종 모델 | 로컬 RTX 4060, 196장·75회 | [best.pt](../models/local_196_75_best.pt) |
| 성능 평가 | 동일 테스트 19장·26객체 | [전체 비교표](../results/test_comparison.csv) |
| 감지 및 추적 | ByteTrack, ID 및 중심점 궤적 | [YouTube 시연](https://youtube.com/shorts/Bvby948NvQ4) |
| 결과 영상 | 전체 2배속 약 6.72초 | [최종 영상](../videos/local_196_75_2x.mp4) |
| 모델 비교 | 로컬·Colab 동일 영상 비교 | [비교 영상](../videos/comparison_2x.mp4) |
| 실험 보고서 | 데이터 구축, 학습, 평가, 영상 분석, 문제 해결 경험 | [보고서](report_1_3_to_9.txt) |
| 실행 코드 | 데이터 검사·학습·평가·영상 추적 | [실행 방법](usage.md) |

공개 저장소: https://github.com/firesmr11/2022041038-PARK-JEONGHWAN-machine-learning-YOLO-fine-tuning
