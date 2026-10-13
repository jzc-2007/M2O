#!/usr/bin/env python3
"""Fetch immutable public assets and verify SHA256 before use."""
import argparse,hashlib,json,os,subprocess,zipfile
from pathlib import Path

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for data in iter(lambda:f.read(8*1024**2),b''):h.update(data)
 return h.hexdigest()
def fetch(url,path,sha):
 path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists() and digest(path)==sha:return
 part=path.with_suffix(path.suffix+'.part')
 subprocess.run(['curl','-fL','--retry','3','--connect-timeout','20','--max-time','7200','-C','-',url,'-o',str(part)],check=True)
 if digest(part)!=sha:raise RuntimeError(f'Checksum mismatch: {path}')
 part.replace(path)
def main():
 p=argparse.ArgumentParser();p.add_argument('--model',choices=['L'],required=True);p.add_argument('--root',type=Path,default=Path('assets'));args=p.parse_args()
 manifest=json.loads((Path(__file__).parent/'assets.lock.json').read_text())
 endpoint=os.environ.get('HF_ENDPOINT','https://huggingface.co').rstrip('/')
 for item in manifest:
  if item.get('model') not in (None,args.model):continue
  url=item.get('url') or f"{endpoint}/{item['repo']}/resolve/{item['revision']}/{item['file']}"
  dest=args.root/item['destination'];fetch(url,dest,item['sha256'])
  if item.get('unzip'):
   target=args.root/item['unzip'];target.mkdir(parents=True,exist_ok=True)
   with zipfile.ZipFile(dest) as z:
    for n in z.namelist():
     if Path(n).is_absolute() or '..' in Path(n).parts:raise ValueError('Unsafe zip member')
    z.extractall(target)
 print('Assets verified:',args.root.resolve())
if __name__=='__main__':main()
