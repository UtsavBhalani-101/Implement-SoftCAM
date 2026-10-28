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

## Experiment 3: ResNet-50 GPU pilot

After converting and validating the RSNA PNG files, run from the repository
root:

```bash
python scripts/run_rsna_training.py \
  --config configs/kaggle_rsna_pilot.yaml \
  --paths configs/kaggle_rsna_full_paths.yaml \
  --require-cuda --print-config
```

The pilot uses one epoch and no ImageNet download (`pretrained: false`). It
checks that the full-size RSNA data, SoftCAM head, validation loop, checkpoint
saving, and TensorBoard logging work on the GPU.

Validate the output before proceeding:

```bash
python scripts/validate_training_artifacts.py \
  --output /kaggle/working/rsna_resnet50_pilot
```

## Experiment 4: paper-scale training

Only after the pilot passes, run the 70-epoch SoftCAM experiment:

```bash
python scripts/run_rsna_training.py \
  --config configs/kaggle_rsna_full.yaml \
  --paths configs/kaggle_rsna_full_paths.yaml \
  --require-cuda --print-config
```

The corresponding non-SoftCAM ResNet-50 baseline is configured in
`configs/rsna_resnet50_full_baseline.yaml` by setting `conv_cls: false`. It
should be run as a separate job with a separate output directory.
