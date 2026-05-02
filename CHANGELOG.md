# Changelog

## [4.0.0] - 2026-05-02

### BREAKING

- Bumped maldet pin to `>=2.0,<3.0` (maldet 2.0 makes `positive_class` mandatory and reorders the binary CM to a fixed `[Benign, Malware]` axis).
- `maldet.toml [output]`: `classes` is now alphabetical `["Benign", "Malware"]` and a new required `positive_class = "Malware"` field declares which label the detector treats as the positive class. Confusion-matrix orientation in lolday's evaluate / predict views derives from these two fields together.
- `maldet.toml [compat]`: `schema_version = 2`, `min_maldet = "2.0"`. Detectors built with maldet < 2.0 are rejected by the lolday cutover-1 backend.
- No on-disk model carry-over: previously trained Lightning checkpoints / `model.joblib` artefacts are not portable across the schema bump because the label encoding now goes through `classes.index(label)` (older artefacts depended on a hard-coded order). Re-train baselines after upgrading.

### Migration

For users embedding elfcnndet:
1. Pin `elfcnndet>=4.0.0` and `maldet[lightning,mlflow]>=2.0,<3.0`.
2. If you fork `maldet.toml`, declare `positive_class` explicitly — the platform refuses to build detectors that omit it under `schema_version = 2`.
3. Re-train any saved baselines; old checkpoints are not compatible.

## [3.0.0] - 2026-04-27

### BREAKING

- Bumped maldet pin to `>=1.1,<2.0`. Detectors built against maldet ≤ 1.0 will not be accepted by lolday phase11e backend (`StageSpec.config_class` and `StageSpec.params_schema` are now required).
- Added Pydantic config classes (`TrainConfig`, `EvaluateConfig`, `PredictConfig`) at `elfcnndet.configs`. `maldet.toml` now references them via `[stages.{stage}].config_class`. `params_schema` is auto-derived at `maldet build` time via `maldet introspect-schema`.

### Migration

For users embedding elfcnndet:
1. Pin `elfcnndet>=3.0.0` and `maldet[lightning,mlflow]>=1.1,<2.0`.
2. Pass typed hyperparameters through the platform UI (lolday phase11e) — JSON Schema is auto-derived; no manual schema upkeep.

## [2.1.0] - 2026-04-27

### Added

- `ByteCNN.predict` and `ByteCNN.predict_proba` so the maldet
  evaluator/predictor (which call the sklearn `.predict`/`.predict_proba`
  API directly) can use a Lightning module without a wrapper. Without
  these, evaluate/predict crashed with
  `AttributeError: 'ByteCNN' object has no attribute 'predict'`.

## [2.0.9] - 2026-04-27

### Fixed

- Pins `maldet[lightning,mlflow]>=1.0.8` so the runner threads
  `model_factory` (the manifest's `[stages.train].model` symbol) into
  `LightningTrainer.load`, fixing evaluate/predict's "requires
  model_factory to rebuild the module" crash.

## [2.0.8] - 2026-04-27

### Fixed

- Pins `maldet[lightning,mlflow]>=1.0.7` so evaluate/predict skip
  samples whose extractor raises `ValueError`. Train had this since
  1.0.1; evaluate/predict were missing the same try/except.

## [2.0.7] - 2026-04-27

### Fixed

- Pulls `maldet[lightning,mlflow]` (was just `lightning`) so the `mlflow`
  package is in the detector image. Without it, `MlflowEventLogger`
  silently no-ops in `log_metric` / `log_artifact`, so platform metrics
  never reach the MLflow tracking server and `runs:/<run_id>/model` is
  empty — exactly what blocked Phase 11d evaluate/predict.

## [2.0.6] - 2026-04-27

### Fixed

- Pins `maldet[lightning]>=1.0.6` so the runner uploads the trained model
  checkpoint to MLflow as `runs:/<run_id>/model` after `trainer.save`.
  Without this, lolday's evaluate/predict model-fetcher init container
  failed to download the source model artifact because the checkpoint
  only ever existed on the train pod's emptyDir.

## [2.0.5] - 2026-04-27

### Fixed

- Pins `maldet[lightning]>=1.0.5` so the LightningTrainer falls back to
  `tempfile.gettempdir()` for checkpoint root when no explicit
  `default_root_dir` is supplied, fixing the v2.0.4 train crash at
  `save_checkpoint` (`OSError [Errno 30] Read-only file system:
  '/app/checkpoints'`) under lolday's `readOnlyRootFilesystem` pod
  security context.

## [2.0.4] - 2026-04-27

### Fixed

- Pins `maldet[lightning]>=1.0.4,<2.0` so pip resolves a torch wheel
  compatible with the lab's NVIDIA driver (560.35.03 / CUDA 12.6). The
  lightning extra in maldet 1.0.4 caps `torch<2.7`; without that ceiling,
  pip pulled torch 2.11 (CUDA 12.8) which crashed at `_cuda_init` with
  "NVIDIA driver too old (found version 12060)".

## [2.0.3] - 2026-04-27

### Fixed

- Pins `maldet[lightning]>=1.0.3,<2.0` (was `>=1.0,<2.0`) so pip cannot
  resolve to an older version. The v2.0.2 build hit a PyPI/CDN propagation
  race and pulled maldet 1.0.2 instead of 1.0.3, which kept the runner on
  the old Hydra-instantiate path and reproduced the "Missing key model"
  failure.

## [2.0.2] - 2026-04-27

### Fixed

- Pulls maldet >=1.0.3 transitively. v1.0.3 changes `StageRunner` to load the
  model class from the manifest's `[stages.train].model` symbol instead of
  Hydra-instantiating from `cfg.model._target_`. Unblocks lolday Phase 11d
  E2E, where the params guard forbids `_target_` overrides, leaving the
  v2.0.0/v2.0.1 train path crashing with `ConfigAttributeError: Missing key
  model`. No source changes here — bump exists to retrigger the lolday build
  pipeline against the patched framework.

## [2.0.1] - 2026-04-27

### Fixed

- Pulls maldet >=1.0.2 transitively, which now skips per-sample feature-extractor
  ValueErrors (e.g. ELF samples lacking `.text` for `Text256Extractor`) instead of
  aborting the whole train run on the first bad sample. No source changes here —
  bump exists to retrigger the lolday build pipeline against the patched framework.

## [2.0.0] - 2026-04-26

### Breaking

- Full rewrite on top of [maldet 1.0](https://pypi.org/project/maldet/) using PyTorch Lightning.
- DDP (Lightning's `strategy="ddp"`) replaces v0's runtime `nn.DataParallel` wrapping.
- Removed: v0 `BaseDetector` ABC, `ElfCnnDetectorConfig` pydantic model, per-detector `elfcnndet` CLI, hand-rolled training loop.
- Dockerfile expects build-time args (`MALDET_*`, `GIT_COMMIT`) emitted as OCI labels required by lolday's detector-build pipeline (image-label contract).

## [0.2.1] - 2026-04-21

Final v0 release on `islab-malware-detector` + `nn.DataParallel`. Deprecated.
