import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from common import audit_dataset


class DatasetAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = self.root/'data.yaml'
        self.config.write_text('names: [cup, mouse, remote]\nnc: 3\ntrain: ../train/images\nval: ../valid/images\ntest: ../test/images\n',encoding='utf-8')
        for split in ['train','valid','test']:
            (self.root/split/'images').mkdir(parents=True)
            (self.root/split/'labels').mkdir()
            (self.root/split/'images/a.jpg').write_bytes(b'audit-content-only')
            (self.root/split/'labels/a.txt').write_text('0 0.5 0.5 0.3 0.4\n1 0.5 0.5 0.2 0.2\n',encoding='utf-8')

    def tearDown(self):
        self.temp.cleanup()

    def test_roboflow_relative_paths_and_object_counts(self):
        normalized, counts, manifests = audit_dataset(self.config)
        self.assertEqual(Path(normalized['train']),self.root/'train/images')
        self.assertEqual(counts['train']['images'],1)
        self.assertEqual(counts['train']['objects'],{'cup':1,'mouse':1,'remote':0})
        self.assertEqual(counts['train']['images_by_class'],{'cup':1,'mouse':1,'remote':0})
        self.assertEqual(len(manifests['test'][0]['label_sha256']),64)

    def test_missing_label_cannot_silently_become_background(self):
        (self.root/'test/labels/a.txt').unlink()
        with self.assertRaises(FileNotFoundError): audit_dataset(self.config)

    def test_nonfinite_box_is_rejected(self):
        (self.root/'train/labels/a.txt').write_text('0 nan 0.5 0.2 0.2\n',encoding='utf-8')
        with self.assertRaises(ValueError): audit_dataset(self.config)

    def test_changed_label_changes_manifest(self):
        first = audit_dataset(self.config)[2]['test'][0]
        (self.root/'test/labels/a.txt').write_text('2 0.5 0.5 0.2 0.2\n',encoding='utf-8')
        second = audit_dataset(self.config)[2]['test'][0]
        self.assertEqual(first['image_sha256'],second['image_sha256'])
        self.assertNotEqual(first['label_sha256'],second['label_sha256'])


if __name__ == '__main__':
    unittest.main()
