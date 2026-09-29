# FPO-YOLO

Official implementation of **FPO-YOLO: A Feature Propagation-Oriented Detector for Small Objects in UAV Imagery**.

FPO-YOLO coordinates high-resolution feature propagation across four components:

- **AAD**: adaptive aggregation downsampling with gated deformable sampling.
- **BCSG**: a bottleneck with channel-spatial residual gating.
- **CIHF**: cross-level interaction and high-resolution fusion.
- **PASF**: position-adaptive scale fusion for the detection head.

The released architecture uses P2, P3, and P4 prediction levels and is implemented on Ultralytics 8.3.237.

## Results

### VisDrone2019-DET validation

| Model | mAP50 | mAP50-95 | Parameters | GFLOPs |
|---|---:|---:|---:|---:|
| YOLO11n | 32.2 | 18.6 | 2.58M | 6.3 |
| FPO-YOLO-n | 40.9 | 24.7 | 1.34M | 13.5 |
| YOLO11s | 38.8 | 23.2 | 9.43M | 21.6 |
| FPO-YOLO-s | **47.9** | **29.2** | **4.13M** | 36.5 |

### VisDrone2019-DET test-dev

| Model | mAP50 | mAP50-95 | Parameters | GFLOPs |
|---|---:|---:|---:|---:|
| YOLO11n | 26.0 | 14.6 | 2.58M | 6.3 |
| FPO-YOLO-n | **33.1** | **19.2** | **1.34M** | 13.5 |
| YOLO11s | 31.4 | 18.0 | 9.43M | 21.6 |
| FPO-YOLO-s | **38.9** | **22.8** | **4.13M** | 36.5 |

## Installation

The paper experiments used Python 3.10, PyTorch 2.4.1, CUDA 11.8, and Ultralytics 8.3.237.

```bash
git clone https://github.com/Azure-127031/FPO-yolo.git
cd FPO-YOLO
pip install -e .
pip install -r requirements-fpo.txt
```

AAD depends on `mmcv.ops.ModulatedDeformConv2d`. Install an MMCV build compatible with your PyTorch and CUDA versions. If a prebuilt wheel is unavailable, follow the official MMCV build instructions.

## Dataset preparation

Download VisDrone2019-DET from the official project and convert its annotations to YOLO format. Arrange the data as follows:

```text
VisDrone2019/
├── images/
│   ├── train/
│   ├── val/
│   └── test-dev/
└── labels/
    ├── train/
    ├── val/
    └── test-dev/
```

Edit `path` in `configs/visdrone.yaml` to point to the dataset root. Dataset files are not redistributed in this repository.

## Training

Train the paper model from a YAML initialization:

```bash
python scripts/train.py --scale s --data configs/visdrone.yaml --device 0
```

The default settings reproduce the paper protocol: 200 epochs, 640-pixel input, batch size 16, SGD, linear learning-rate decay, seed 0, AMP, and Mosaic disabled for the final 10 epochs.

## Validation

```bash
python scripts/val.py --weights path/to/best.pt --data configs/visdrone.yaml --device 0
```

Model selection must use the validation set. Evaluate the selected `best.pt` checkpoint on test-dev only after model selection.

## Model definitions

- `ultralytics/cfg/models/11/yolo11-fpo.yaml`: FPO-YOLO n/s/m/l/x configurations.
- `ultralytics/nn/modules/fpo_aad.py`: AAD.
- `ultralytics/nn/modules/fpo_bcsg.py`: BCSG.
- `ultralytics/nn/modules/fpo_cihf.py`: CIHF/SBA.
- `ultralytics/nn/modules/fpo_pasf.py`: PASF detection head.

## Reproducibility notes

- The reported models are initialized from YAML unless an experiment explicitly states otherwise.
- Keep dataset splits, image size, epochs, augmentation, optimizer, and evaluation settings identical when comparing with a baseline.
- Report preprocessing, inference, and postprocessing latency separately when possible; postprocessing time depends on prediction density.
- Checkpoints and additional evaluation artifacts will be added after final verification.

## License

This repository is derived from Ultralytics 8.3.237 and is distributed under the [AGPL-3.0 License](LICENSE). See the upstream [Ultralytics repository](https://github.com/ultralytics/ultralytics) for the original project and documentation.

## Citation

The manuscript is currently under review. Citation metadata will be added after publication.
