#!/usr/bin/env python3
"""Compile the self-contained arXiv source in an isolated build directory."""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess,tempfile,zipfile
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=root.parent/'m2o-release-artifacts');args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
source=root/'paper'
files=[x for x in source.rglob('*') if x.is_file() and x.suffix in ['.tex','.bib','.bbl','.sty','.bst','.pdf']]
with tempfile.TemporaryDirectory(prefix='m2o-arxiv-') as d:
 build=Path(d)
 for f in files:
  target=build/f.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,target)
 def run(cmd):
  result=subprocess.run(cmd,cwd=build,capture_output=True,text=True)
  if result.returncode:raise RuntimeError(result.stdout[-5000:]+result.stderr[-2000:])
 latex=['pdflatex','-interaction=nonstopmode','-halt-on-error','-file-line-error','main.tex']
 run(latex);run(['bibtex','main'])
 for _ in range(3):run(latex)
 log=(build/'main.log').read_text()
 for failure in ['LaTeX Error:','undefined references','Citation `','Rerun to get cross-references right','Overfull \\hbox']:
  assert failure not in log,failure
 shutil.copy2(build/'main.bbl',source/'main.bbl')
 shutil.copy2(build/'main.pdf',args.output/'m2o_arxiv.pdf')
 shutil.copy2(build/'main.aux',args.output/'build.aux')
 with zipfile.ZipFile(args.output/'m2o_arxiv_source.zip','w',zipfile.ZIP_DEFLATED) as z:
  for f in files:z.write(source/f.relative_to(source),str(f.relative_to(source)))
 (args.output/'build_report.json').write_text(json.dumps({'source_files':len(files),'source_sha256':{str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},'local_compile_passed':True,'arxiv_uploaded':False},indent=2))
 print(args.output/'m2o_arxiv.pdf')
