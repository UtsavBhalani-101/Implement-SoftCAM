# Initial experiments

## Experiment 1: local CPU smoke test

From the `softcam` directory:

```bash
python scripts/cpu_smoke_test.py
```

This uses a tiny synthetic backbone and verifies the official `FullyConvModel`
head, evidence-map shape, ElasticNet-style penalty, backward pass, and optimizer
step. It does not require a dataset, CUDA, pretrained weights, or network access.

## Experiment 2: short Kaggle training run

Attach the repository and the RSNA image dataset to a Kaggle notebook. Update
`configs/kaggle_short_paths.yaml` if the mounted dataset path differs, then run:

```bash
python scripts/run_kaggle_short.py
```

The run trains one epoch with batch size 4 and writes checkpoints and logs to
`/kaggle/working/softcam_short_rsna`. This is a pipeline check, not a paper
result. Save that directory as a Kaggle output before the session ends.
