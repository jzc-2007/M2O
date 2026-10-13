# Release validation

The public configuration is `configs/l2.yml`. It preserves the paper's L/2
four-step-teacher, batch-4, 16K-update recipe and the original evaluation bank.
The scientific trainer, evaluator and feature objective are unchanged.

The validated Linux environment is recorded in `environment-a100-20261011.txt`.
The fresh L/2 run completed 16K updates, with 642 finite metric records and
full-state restoration from its smoke checkpoint. Its measured A100 cost is
separate from the original H200 result. Reports and source provenance are in
`results/` and `docs/CODE_PROVENANCE.json`.

The L/2 full-state archive passed SHA256, zstd integrity, safe-path, Orbax
manifest and required state-field checks. The original H200 adapted optimizer
state was not recovered; the A100 archive is a fresh reproduction.

The paper matches the pre-extension manuscript except for the additional
Table 4 row and a local table-row-spacing adjustment. It has nine main-text
pages and eighteen total. References resolve, fonts are embedded and the
source ZIP contains all required figures, bibliography and style files.

The source package excludes credentials, private machine addresses, run
directories, deployment controllers and unselected experiment configurations.
Weights are distributed separately. Public code and weights are distributed via the GitHub repository and its release.

The narrowed source package passed all 12 CPU tests in 3.18 seconds on the
validated runtime with GPU visibility disabled. Python compilation also passed.
All 18 manuscript pages were rendered and visually checked.
