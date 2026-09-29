# FPO-YOLO

这是论文 **FPO-YOLO: A Feature Propagation-Oriented Detector for Small Objects in UAV Imagery** 的官方代码实现。

FPO-YOLO 面向无人机图像小目标检测，在 P2、P3 和 P4 三个预测尺度上协调四个模块：

- **AAD**：在降采样前执行门控可变形采样；
- **BCSG**：结合结构响应与空间、通道残差门控；
- **CIHF**：进行跨层交互并构建高分辨率融合特征；
- **PASF**：在检测头中按空间位置自适应选择尺度特征。

## 安装

论文实验基于 Python 3.10、PyTorch 2.4.1、CUDA 11.8 和 Ultralytics 8.3.237。

```bash
git clone https://github.com/Azure-127031/FPO-yolo.git
cd FPO-YOLO
pip install -e .
pip install -r requirements-fpo.txt
```

AAD 使用 `mmcv.ops.ModulatedDeformConv2d`，需要安装与本机 PyTorch 和 CUDA 对应的 MMCV。

## 数据集

下载 VisDrone2019-DET，并转换为 YOLO 标注格式。修改 `configs/visdrone.yaml` 中的 `path`，使其指向数据集根目录。仓库不包含数据集文件。

## 训练

```bash
python scripts/train.py --scale s --data configs/visdrone.yaml --device 0
```

默认训练设置与论文一致：200 轮、输入尺寸 640、batch 16、SGD、线性学习率衰减、随机种子 0、AMP，并在最后 10 轮关闭 Mosaic。

## 验证

```bash
python scripts/val.py --weights path/to/best.pt --data configs/visdrone.yaml --device 0
```

应先在验证集上选择 `best.pt`，再使用确定的权重评估 test-dev。

## 代码位置

- `ultralytics/cfg/models/11/yolo11-fpo.yaml`：FPO-YOLO n/s/m/l/x 配置；
- `ultralytics/nn/modules/fpo_aad.py`：AAD；
- `ultralytics/nn/modules/fpo_bcsg.py`：BCSG；
- `ultralytics/nn/modules/fpo_cihf.py`：CIHF/SBA；
- `ultralytics/nn/modules/fpo_pasf.py`：PASF 检测头。

## 许可

本仓库基于 Ultralytics 8.3.237，遵循 [AGPL-3.0 License](LICENSE)。论文仍在投稿阶段，正式发表后将补充引用信息。
