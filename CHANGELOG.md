# Changelog

## [2.0.0] - 2026-04-26

### Breaking

- Full rewrite on top of [maldet 1.0](https://pypi.org/project/maldet/) using PyTorch Lightning.
- DDP (Lightning's `strategy="ddp"`) replaces v0's runtime `nn.DataParallel` wrapping.
- Removed: v0 `BaseDetector` ABC, `ElfCnnDetectorConfig` pydantic model, per-detector `elfcnndet` CLI, hand-rolled training loop.
- Dockerfile expects build-time args (`MALDET_*`, `GIT_COMMIT`) emitted as OCI labels for lolday Phase 11c's pipeline.

## [0.2.1] - 2026-(prior)

Final v0 release on `islab-malware-detector` + `nn.DataParallel`. Deprecated.
