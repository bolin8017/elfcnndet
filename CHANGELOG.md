# Changelog

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
