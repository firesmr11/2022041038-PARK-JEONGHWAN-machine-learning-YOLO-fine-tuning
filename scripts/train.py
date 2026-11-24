"""Independent YOLOv8n training from pretrained weights; validation selects best.pt."""
import argparse
import platform
import time
from pathlib import Path
from common import audit_dataset, portable_yaml, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--epochs', type=int, choices=[25, 50, 75], default=50)
    p.add_argument('--device', default='0', help='0 for CUDA, cpu for explicit CPU execution')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--project', type=Path, default=Path('runs/train'))
    p.add_argument('--name')
    a = p.parse_args()
    a.project = a.project.resolve()
    import torch
    import ultralytics
    from ultralytics import YOLO
    if ultralytics.__version__ != '8.4.216':
        raise RuntimeError('Install ultralytics==8.4.216 for this experiment')
    if a.device != 'cpu' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Install a CUDA-enabled PyTorch build or explicitly choose --device cpu.')
    config, counts, manifests = audit_dataset(a.data)
    a.project.mkdir(parents=True, exist_ok=True)
    name = a.name or f'train{counts["train"]["images"]}_epochs{a.epochs}'
    # Preserve the original Roboflow YAML and resolve paths for this machine only.
    config_file = portable_yaml(config, a.project / 'dataset_configs' / f'{name}.yaml')
    model = YOLO('yolov8n.pt')
    start = time.perf_counter()
    model.train(data=str(config_file), epochs=a.epochs, imgsz=640, batch=16,
                device=a.device, workers=a.workers, seed=0, deterministic=True,
                patience=0, optimizer='auto', amp=True, plots=True, save=True,
                project=str(a.project), name=name, exist_ok=False)
    run = Path(model.trainer.save_dir)
    write_json(run / 'experiment.json', {'requested_epochs': a.epochs, 'counts': counts,
               'train_call_wall_seconds': time.perf_counter() - start,
               'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                               'torch': torch.__version__, 'ultralytics': ultralytics.__version__,
                               'device': a.device,
                               'gpu': torch.cuda.get_device_name(0) if a.device != 'cpu' else None}})
    write_json(run / 'dataset_manifest.json', manifests)
    print('Training complete. Evaluate separately:', run / 'weights/best.pt')


if __name__ == '__main__':
    main()
