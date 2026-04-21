# elfcnndet

Small 1D-CNN malware detector for Linux ELF binaries. Feature = first 256
bytes of the `.text` section, fed through a byte-embedding + conv stack.

Designed as a **multi-GPU template** for the
[lolday](https://github.com/louiskyee/lolday) platform — detects
`torch.cuda.device_count()` at runtime and wraps the model in
`nn.DataParallel` when 2 GPUs are allocated.

Inherits [islab-malware-detector](https://github.com/bolin8017/islab-malware-detector)
`BaseDetector`.

## Usage

```bash
elfcnndet init --output config.json
elfcnndet train    --config config.json
elfcnndet evaluate --config config.json
elfcnndet predict  --config config.json
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

## GPU scheduling behaviour

```python
device_count = torch.cuda.device_count()
if device_count >= 2:
    model = nn.DataParallel(model)  # splits each batch across both GPUs
```

MLflow run tags include `gpu_device_count` so the lolday UI / MLflow UI
surfaces whether a given run actually used both cards.

On lolday, pass `resource_profile="gpu2"` when submitting the job:

```json
POST /api/v1/jobs
{
  "type": "train",
  "detector_version_id": "...",
  "train_dataset_id": "...",
  "resource_profile": "gpu2"
}
```

## Dataset format

Same as upxelfdet / elfrfdet: CSV with `file_name,label[,family]`,
samples at `<dataset_root>/<sha[:2]>/<sha>`.

## License

MIT
