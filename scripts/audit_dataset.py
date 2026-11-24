"""Audit labels and compare the exact validation/test content of two exports."""
import argparse
from collections import Counter
from pathlib import Path
from common import audit_dataset, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--compare', type=Path)
    p.add_argument('--out', type=Path, default=Path('runs/dataset_audit.json'))
    a = p.parse_args()
    _, counts, manifests = audit_dataset(a.data)
    result = {'counts': counts, 'manifest': manifests}
    if a.compare:
        _, other_counts, other = audit_dataset(a.compare)
        result['other_counts'] = other_counts
        def content(rows):
            return Counter((r['image_sha256'], r['label_sha256']) for r in rows)
        result['identical_eval_content'] = {s: content(manifests[s]) == content(other[s]) for s in ['val', 'test']}
        result['first_train_subset_of_second'] = not (content(manifests['train']) - content(other['train']))
        write_json(a.out, result)
        if not all(result['identical_eval_content'].values()):
            raise ValueError('Validation/test content differs. Do not compare as fixed-split experiments.')
    else:
        write_json(a.out, result)
    print(counts)
    print('Saved:', a.out)


if __name__ == '__main__':
    main()
