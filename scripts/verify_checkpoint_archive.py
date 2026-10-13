#!/usr/bin/env python3
"""Verify an immutable full-state archive against its published SHA256."""
import argparse,hashlib,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('--sha256',required=True);a=p.parse_args()
h=hashlib.sha256()
with a.archive.open('rb') as f:
 for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
if h.hexdigest()!=a.sha256:raise SystemExit('SHA256 mismatch; preserve partial file and resume download')
subprocess.run(['zstd','-t',str(a.archive)],check=True)
entries=subprocess.check_output(['tar','-tf',str(a.archive)],text=True).splitlines()
if any(Path(e).is_absolute() or '..' in Path(e).parts for e in entries):raise SystemExit('Unsafe archive path')
if not any(e.endswith('manifest.ocdbt') for e in entries):raise SystemExit('Checkpoint manifest missing')
metadata_paths=[e for e in entries if e.endswith('/_METADATA')]
if len(metadata_paths)!=1:raise SystemExit('Expected one training-state metadata file')
metadata=json.loads(subprocess.check_output(['tar','-xOf',str(a.archive),metadata_paths[0]],text=True))
fields={v['key_metadata'][0]['key'] for v in metadata['tree_metadata'].values()}
required={'step','params','ema_params','opt_state','rms_sq'}
if not required.issubset(fields):raise SystemExit('Incomplete training state: '+str(required-fields))
print(json.dumps({'state_fields':sorted(fields),'archive':a.archive.name,'bytes':a.archive.stat().st_size,'sha256':h.hexdigest(),'entries':len(entries),'verified':True}))
