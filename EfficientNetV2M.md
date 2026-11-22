# EfficientNetV2M 缺陷分类复现

沿用同一项目中 InceptionResNetV2 的 Python 环境、数据和逐图划分，比较三组数据上的 Adam、SGD、RMSprop。结果使用最终一轮模型，主指标为测试集判对图片数除以测试图片总数。验证准确率单独记录。

## 运行与复核

在本文件所在目录运行。现有环境和权重支持离线重跑。

```bash
# 全部九次训练
.venv/bin/python train.py --model EfficientNetV2M

# 或单独训练一组
.venv/bin/python train.py --model EfficientNetV2M --study three --optimizer SGD

# 从逐图概率复算九组准确率并写入汇总表
.venv/bin/python verify.py --model EfficientNetV2M
```

在另一台电脑上，先按 [基础环境与数据准备](README.md) 安装 requirements.txt 并下载三组数据。脚本第一次加载 EfficientNetV2M 时会通过 Keras 下载对应的官方无顶层 ImageNet 权重。也可预先将 efficientnetv2-m_notop.h5 放入 .keras/models，其 MD5 为 1b8dc055df72dde80d614482840fe342。

## 实验设置

- 输入为 OpenCV 读取的 224×224 RGB 图像。沿用作者 notebook 的外部像素除以 255，同时保留 Keras EfficientNetV2M 默认的内置预处理。在本次 Keras 3.8.0 中，该层执行 x/128−1，主干实际输入范围为 [−1, −0.9921875]。这是解释结果时需要考虑的实验条件。
- 冻结 ImageNet 主干，BatchNorm 采用推理模式。保留作者的 4×4 平均池化、Flatten、64 单元 ReLU、Dropout(0.5) 和双输出 Softmax 分类头，损失为 binary_crossentropy。缓存每张图片至 Flatten 的 1280 维特征，训练前用首批图片验证完整网络与缓存路径的分类概率一致。
- 每个数据组内，三种优化器使用相同初始化、随机种子 42 和学习率 0.001，分别独立训练。Study One、Two、Three 的批大小依次为 8、50、30，训练轮数依次为 30、50、30。作者 notebook 靠手动切换优化器，本实现按命令行参数选择并记录实际优化器。
- 按路径排序后，使用作者的两次随机划分。Study One 的公开原图有 136 张缺陷和 135 张正常图，比论文表 1 少一张正常图。作者未公开逐图划分，具体测试样本可能有差异。Study Two 使用作者已增强的图像，按图像划分测试集。

[作者训练 notebook](https://github.com/mahsan2/AM3d/blob/main/basic_algorithm.ipynb) · [作者数据集](https://github.com/mahsan2/AMdataset) · [Study Three 数据集](https://www.kaggle.com/datasets/justin900429/3d-printer-defected-dataset) · [论文表 2–10](https://doi.org/10.1016/j.eswa.2024.124293)

## 结果文件

[九组准确率汇总](results_efficientnetv2m/summary.csv) 给出验证准确率、测试正确数、测试总数、测试准确率、论文对照值和相差百分点。各组的 split.csv 记录逐图划分。每个优化器目录均包含逐图预测、每轮训练日志、run.json 配置和最终分类头权重。

## 本次实测

| 数据组 | Adam 测试准确率 | SGD 测试准确率 | RMSprop 测试准确率 | 每个优化器判对 / 总数 | 论文测试准确率 |
|---|---:|---:|---:|---:|---:|
| Study One | 49.09% | 49.09% | 49.09% | 27 / 55 | 49% |
| Study Two | 66.89% | 66.89% | 66.89% | 400 / 598 | 67% |
| Study Three | 50.49% | 50.49% | 50.49% | 156 / 309 | 50% |

九次运行都把全部测试图片预测为正常。三组测试中的缺陷图片分别为 28、198 和 153 张，检出数均为零，缺陷召回率均为 0%。因此，表中准确率等于相应测试集的正常图比例。输入的两次缩放是一个可能原因。本次实验未单独改变缩放方式检验其影响。[Keras 输入约定](https://keras.io/api/applications/efficientnet_v2/efficientnet_v2_models/#efficientnetv2m-function)
