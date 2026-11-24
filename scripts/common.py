"""Portable dataset paths, audits and machine-readable experiment records."""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

CLASSES = ['cup', 'mouse', 'remote']
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def audit_dataset(data_file):
    import yaml
    data_file = Path(data_file).resolve()
    config = yaml.safe_load(data_file.read_text(encoding='utf-8-sig'))
    names = config['names']
    if isinstance(names, dict):
        names = [names.get(i, names.get(str(i))) for i in range(len(names))]
    if names != CLASSES:
        raise ValueError(f'Expected {CLASSES}, got {names}')
    root = data_file.parent
    configured_root = Path(config.get('path') or root)
    if not configured_root.is_absolute():
        configured_root = root / configured_root
    normalized, counts, manifests = dict(config), {}, {}
    normalized['names'] = CLASSES
    normalized['nc'] = 3
    for key, folder in [('train', 'train'), ('val', 'valid'), ('test', 'test')]:
        value = config.get(key)
        if not isinstance(value, str):
            raise ValueError(f'{key} must specify a directory in this dataset export')
        configured = Path(value)
        candidates = [configured_root / configured, root / configured,
                      root / folder / 'images', root / key / 'images']
        images_dir = next((p.resolve() for p in candidates if p.is_dir()), None)
        if images_dir is None:
            raise FileNotFoundError(f'Cannot resolve {key} images from {data_file}')
        images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in EXTENSIONS)
        if not images:
            raise ValueError(f'No images in {images_dir}')
        objects, image_classes = Counter(), Counter()
        empty = 0
        records = []
        for image in images:
            label = images_dir.parent / 'labels' / (image.stem + '.txt')
            if not label.is_file():
                raise FileNotFoundError(f'Missing label: {label}')
            lines = [r.split() for r in label.read_text(encoding='utf-8-sig').splitlines() if r.strip()]
            empty += not lines
            present = set()
            for row in lines:
                if len(row) != 5:
                    raise ValueError(f'Expected class cx cy w h: {label}')
                cls, box = int(row[0]), list(map(float, row[1:]))
                if cls not in range(3) or not all(math.isfinite(v) and 0 <= v <= 1 for v in box) or min(box[2:]) <= 0:
                    raise ValueError(f'Invalid box in {label}: {row}')
                objects[cls] += 1
                present.add(cls)
            image_classes.update(present)
            records.append({'image': image.name, 'image_sha256': sha256(image), 'label_sha256': sha256(label)})
        counts[key] = {'images': len(images), 'empty_labels': empty,
                       'objects': {CLASSES[i]: objects[i] for i in range(3)},
                       'images_by_class': {CLASSES[i]: image_classes[i] for i in range(3)}}
        manifests[key] = records
        normalized[key] = str(images_dir)
    normalized['path'] = str(root)
    # Dataset download URLs are not executed; exported files must already be present.
    normalized.pop('download', None)
    return normalized, counts, manifests


def portable_yaml(config, target):
    import yaml
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf-8')
    return target.resolve()


def metrics_summary(metrics, names):
    evaluated = set(int(i) for i in metrics.box.ap_class_index)
    return {'Precision_percent': float(metrics.box.mp) * 100,
            'Recall_percent': float(metrics.box.mr) * 100,
            'mAP50_percent': float(metrics.box.map50) * 100,
            'mAP50_95_percent': float(metrics.box.map) * 100,
            'per_class_mAP50_95_percent': {names[i]: float(metrics.box.maps[i]) * 100 for i in evaluated}}
