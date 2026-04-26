# elfcnndet

1D-CNN malware detector for Linux ELF binaries with PyTorch Lightning. Feature = first 256 bytes of the `.text` section, fed through a byte-embedding + conv stack. Multi-GPU training via Lightning **DDP** (replaces the v0 `nn.DataParallel` pattern). Reference template for the [lolday](https://github.com/louiskyee/lolday) platform on the [maldet 1.0](https://github.com/bolin8017/maldet) framework.

## Install

```bash
pip install -e .[dev]
maldet check
maldet describe
```

## CLI

```bash
maldet run train    --config config.yaml
maldet run evaluate --config config.yaml
maldet run predict  --config config.yaml
```

## Architecture

```
input bytes (N, 256) uint8
    ↓ nn.Embedding(256 → 32)
    ↓ Conv1d(32 → 64, kernel=5) + ReLU + MaxPool1d(2)
    ↓ Conv1d(64 → 128, kernel=3) + ReLU
    ↓ AdaptiveAvgPool1d(1)
    ↓ Linear(128 → 2)
    ↓ CrossEntropyLoss
```

`ByteCNN(LightningModule)` in `src/elfcnndet/models.py`.

## Distributed training (DDP)

`maldet.toml` declares `lifecycle.supports_distributed = "ddp"`. When lolday's backend submits a `gpu2` resource profile, `maldet.trainers.lightning_trainer` reads `MALDET_GPU_COUNT=2` + `MALDET_DISTRIBUTED_STRATEGY=ddp` from the env and configures `Trainer(strategy="ddp", devices=2)`. No more hand-rolled `nn.DataParallel`.

## On lolday

1. Register: `POST /api/v1/detectors { git_url: "https://github.com/bolin8017/elfcnndet.git" }`.
2. Build a tag: `POST /api/v1/detectors/{id}/builds { git_tag: "v2.0.0" }`.
3. Submit a job: `POST /api/v1/jobs { type: "train", resource_profile: "gpu2", ... }`. Phase 11b's `validate_job_submission` permits the multi-GPU profile because `manifest.lifecycle.supports_distributed = "ddp"`.

## Migrating from v0.2.x

v2 is a full rewrite. v0 `BaseDetector`, `ElfCnnDetectorConfig`, per-detector CLI, and runtime `nn.DataParallel` wrapping are all removed. Use `maldet run <stage>` and let Lightning's `strategy="ddp"` handle multi-GPU.

## License

MIT
