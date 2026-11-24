"""Evaluate a saved checkpoint on the test split, without retraining."""
import argparse
from pathlib import Path
from common import audit_dataset, metrics_summary, portable_yaml, sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--project', type=Path, default=Path('runs/evaluate'))
    p.add_argument('--name', default='test')
    a = p.parse_args()
    a.project = a.project.resolve()
    if not a.model.is_file():
        p.error('Model does not exist')
    from ultralytics import YOLO
    config, counts, _ = audit_dataset(a.data)
    data_file = portable_yaml(config, a.project / 'dataset_configs' / f'{a.name}.yaml')
    model = YOLO(str(a.model))
    saved_dirs = []
    model.add_callback('on_val_end', lambda validator: saved_dirs.append(Path(validator.save_dir)))
    metrics = model.val(data=str(data_file), split='test', imgsz=640, batch=16,
                        device=a.device, workers=2, plots=True,
                        project=str(a.project), name=a.name, exist_ok=False)
    result = {'model_sha256': sha256(a.model), 'split': 'test', 'test_counts': counts['test'],
              **metrics_summary(metrics, model.names)}
    run = saved_dirs[-1]
    write_json(run / 'test_summary.json', result)
    print(result)
    print('Saved:', run / 'test_summary.json')


if __name__ == '__main__':
    main()
