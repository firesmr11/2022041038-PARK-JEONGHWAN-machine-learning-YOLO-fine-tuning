"""Verify checked-in model bytes, comparison rows, and fully decoded video lengths."""
import csv
import json
from pathlib import Path
from common import sha256


def main():
    root = Path(__file__).resolve().parents[1]
    inventory = json.loads((root/'results/checkpoint_inventory.json').read_text(encoding='utf-8'))
    assert len(inventory) == 7, 'Expected 7 provided checkpoints'
    for item in inventory:
        path = root / item['file']
        assert path.stat().st_size == item['bytes'], f'Wrong size: {path}'
        assert sha256(path) == item['sha256'], f'Wrong hash: {path}'
    with (root/'results/test_comparison.csv').open(encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    combinations = {(r['environment'],int(r['train_images']),int(r['epochs'])) for r in rows}
    assert len(rows) == len(combinations) == 12
    assert {r['environment'] for r in rows} == {'Colab T4','Local RTX 4060'}
    assert {int(r['train_images']) for r in rows} == {196,414}
    assert {int(r['epochs']) for r in rows} == {25,50,75}
    for row in rows:
        for key in ['Precision_percent','Recall_percent','mAP50_percent','mAP50_95_percent']:
            assert 0 <= float(row[key]) <= 100
    import cv2
    video_report = []
    files = list((root/'videos').glob('*.mp4')) + list((root/'videos/original').glob('*.mp4'))
    assert len(files) == 5
    for path in sorted(files):
        cap = cv2.VideoCapture(str(path))
        assert cap.isOpened(), path
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = 0
        while True:
            ok, _ = cap.read()
            if not ok: break
            count += 1
        cap.release()
        assert count == 403, (path,count)
        assert abs(fps-(60 if path.stem.endswith('_2x') else 30)) < .01, (path,fps)
        video_report.append({'file':path.relative_to(root).as_posix(),'decoded_frames':count,'fps':fps,'duration_seconds':count/fps})
    required = ['README.md','data/README.md','docs/report_1_3_to_9.txt','docs/submission_checklist.md',
                'notebooks/colab_experiments.ipynb','results/figures/test_map_comparison.png','scripts/compare_video.py']
    assert all((root/p).is_file() for p in required)
    print(json.dumps({'models_verified':len(inventory),'test_rows_verified':len(rows),'videos':video_report},indent=2))


if __name__ == '__main__':
    main()
