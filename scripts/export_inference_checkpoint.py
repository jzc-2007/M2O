#!/usr/bin/env python3
"""Wrap raw Flax inference weights; does not recreate optimizer state."""
import argparse,json,hashlib
from pathlib import Path
import flax.serialization
from flax.training import checkpoints
p=argparse.ArgumentParser();p.add_argument('--raw-msgpack',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--step',type=int,required=True);args=p.parse_args()
if args.output.exists():raise ValueError('Choose a new output directory')
b=args.raw_msgpack.read_bytes();params=flax.serialization.msgpack_restore(b)
args.output.mkdir(parents=True)
checkpoints.save_checkpoint(str(args.output.resolve()),{'params':params,'step':args.step},args.step)
(args.output/'INFERENCE_ONLY.json').write_text(json.dumps({'step':args.step,'raw_sha256':hashlib.sha256(b).hexdigest(),'optimizer_state':False,'training_resume_supported':False},indent=2))
