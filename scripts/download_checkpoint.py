#!/usr/bin/env python3
"""Download release shards, verify them, and assemble one checkpoint archive."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def valid(path, spec):
    return (path.is_file() and path.stat().st_size == spec['bytes']
            and digest(path) == spec['sha256'])


def download(spec, directory):
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / spec['archive']
    if archive.exists():
        if valid(archive, spec):
            return archive
        raise ValueError(f'Existing archive fails verification: {archive}; move it before retrying')
    for part in spec['parts']:
        target = directory / part['name']
        if valid(target, part):
            continue
        if target.exists():
            raise ValueError(f'Existing shard fails verification: {target}; move it before retrying')
        partial = target.with_name(target.name + '.partial')
        if partial.exists() and partial.stat().st_size >= part['bytes']:
            partial.unlink()
        print(f"Downloading {part['name']}", flush=True)
        subprocess.run(['curl', '--fail', '--location', '--retry', '3',
                        '--connect-timeout', '30', '--continue-at', '-',
                        '--output', str(partial), part['url']], check=True)
        if not valid(partial, part):
            partial.unlink()
            raise ValueError(f"Shard checksum mismatch: {part['name']}; rerun to retry")
        partial.replace(target)
    staging = archive.with_name(archive.name + '.assembling')
    with staging.open('wb') as output:
        for part in spec['parts']:
            with (directory / part['name']).open('rb') as source:
                shutil.copyfileobj(source, output, 8 * 1024 * 1024)
    if not valid(staging, spec):
        staging.unlink()
        raise ValueError('Assembled archive checksum mismatch')
    staging.replace(archive)
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['L', 'XL'], required=True)
    parser.add_argument('--output', type=Path, default=Path('checkpoints'))
    args = parser.parse_args()
    manifest = json.loads((Path(__file__).parent / 'checkpoints.lock.json').read_text())
    archive = download(manifest['models'][args.model], args.output / args.model.lower())
    print(f'Verified archive: {archive}')
    print('Extract it to a new directory; the checkpoint is checkpoint_16000/.')
    print('After verification, the .partNN files may be deleted to reclaim space.')


if __name__ == '__main__':
    main()
