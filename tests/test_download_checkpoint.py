"""Check download resumption, shard integrity, and safe assembly without GPUs."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('download_checkpoint', Path(__file__).parents[1] / 'scripts/download_checkpoint.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class CheckpointDownloadTests(unittest.TestCase):
    def fixture(self, root):
        contents = [b'first checkpoint shard' * 32, b'second shard' * 17]
        parts = []
        for i, data in enumerate(contents):
            source = root / f'source{i}'
            source.write_bytes(data)
            parts.append({'name': f'weights.part{i:02d}', 'bytes': len(data),
                          'sha256': hashlib.sha256(data).hexdigest(), 'url': source.as_uri()})
        joined = b''.join(contents)
        return {'archive': 'weights.zst', 'bytes': len(joined),
                'sha256': hashlib.sha256(joined).hexdigest(), 'parts': parts}, contents

    def test_resume_and_reuse_verified_archive(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); recipe, chunks = self.fixture(root); out = root / 'out'; out.mkdir()
            (out / 'weights.part00.partial').write_bytes(chunks[0][:30])
            result = module.download(recipe, out)
            self.assertEqual(result.read_bytes(), b''.join(chunks))
            for p in root.glob('source*'): p.unlink()
            self.assertEqual(module.download(recipe, out), result)

    def test_reject_corrupt_existing_shard_without_overwriting(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); recipe, _ = self.fixture(root); out = root / 'out'; out.mkdir()
            p = out / 'weights.part00'; p.write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError, 'Existing shard'):
                module.download(recipe, out)
            self.assertEqual(p.read_bytes(), b'corrupted')
            self.assertFalse((out / 'weights.zst').exists())

    def test_reject_bad_archive_checksum(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); recipe, _ = self.fixture(root); recipe['sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'Assembled archive'):
                module.download(recipe, root / 'out')
            self.assertFalse((root / 'out/weights.zst').exists())
            self.assertFalse((root / 'out/weights.zst.assembling').exists())

if __name__ == '__main__':
    unittest.main()
