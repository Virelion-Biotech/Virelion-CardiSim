import hashlib
import json
import zipfile

from scripts.export_colab_results import export


def test_repeated_export_has_one_nonrecursive_manifest_and_verified_payloads(tmp_path):
    root = tmp_path / 'run'
    (root / 'results').mkdir(parents=True)
    (root / 'state').mkdir()
    (root / 'data').mkdir()
    (root / 'results' / 'metrics.json').write_text('{"rmse":8.78}')
    (root / 'state' / 'model.pt').write_bytes(b'weights-stay-on-drive')
    (root / 'data' / 'raw.pt').write_bytes(b'raw-data-stays-on-drive')
    destination = tmp_path / 'results.zip'
    for _ in range(2):
        export(root, destination)
        with zipfile.ZipFile(destination) as archive:
            names = archive.namelist()
            assert names.count('ARTIFACT_HASHES.json') == 1
            manifest = json.loads(archive.read('ARTIFACT_HASHES.json'))
            assert 'ARTIFACT_HASHES.json' not in manifest
            assert set(names) == set(manifest) | {'ARTIFACT_HASHES.json'}
            for name, digest in manifest.items():
                assert hashlib.sha256(archive.read(name)).hexdigest() == digest
            assert 'state/model.pt' not in names
            assert 'data/raw.pt' not in names
            retained = json.loads(archive.read('RETAINED_ON_DRIVE.json'))
            assert retained['state/model.pt']['bytes'] == len(b'weights-stay-on-drive')
