# Checkpoints

[Download release v1.0.0](https://github.com/jzc-2007/M2O/releases/tag/v1.0.0).
Weights are separate from source. Archives use Orbax and preserve `params`,
`ema_params`, `opt_state`, `step`, and `rms_sq`. Use raw parameters for the
reported results.

| Model | Download command | Archive SHA256 |
|---|---|---|
| L/2 | `python scripts/download_checkpoint.py --model L` | `d6d0da7ea2271471c0c4fb61fd9e94b508e86d82224370de1551cfca2f6bb157` |
| XL/2 (FID-50K 1.62) | `python scripts/download_checkpoint.py --model XL` | `d736004e833dba542d59e22dcdaa1f6f4aa3019a50e0e02a88d5f01c8659188b` |

GitHub limits each release asset to less than 2 GiB. The downloader retrieves
1 GiB shards, resumes interrupted downloads, and verifies each shard and the
merged archive against `scripts/checkpoints.lock.json`. It requires Python 3
and `curl`, without model dependencies. Allow approximately twice the archive
size for the shards and merged file. After successful verification, shard files
can be removed.

The commands produce `checkpoints/l/m2o_l2_checkpoint.tar.zst` or
`checkpoints/xl/m2o_xl_checkpoint.tar.zst`. Each archive contains only one
`checkpoint_16000/` directory, with no run configuration, logs or reports.
Install `zstd` to check and extract archives:

```bash
python scripts/verify_checkpoint_archive.py checkpoints/l/m2o_l2_checkpoint.tar.zst --sha256 d6d0da7ea2271471c0c4fb61fd9e94b508e86d82224370de1551cfca2f6bb157
mkdir -p checkpoints/l/extracted
tar --zstd -xf checkpoints/l/m2o_l2_checkpoint.tar.zst -C checkpoints/l/extracted
```

For L/2 evaluation, use the README command with
`--checkpoint-dir checkpoints/l/extracted/checkpoint_16000`.
The L/2 checkpoint is a fresh A100 reproduction, separate from the paper's
original H200 run. Its state can resume the L/2 recipe in `configs/l2.yml`.
Keep the configuration unchanged when resuming. Environment versions are
recorded in `environment-a100-20261011.txt`.

Manual downloads use the ordered `.part01`, `.part02`, ... assets from the
release. Concatenate only one model's parts in that order. The release's
`SHA256SUMS` includes every shard and each complete archive. Check archive
integrity and required state fields before use.
