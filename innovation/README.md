# MSAF-EfficientNetV2M：自适应多尺度特征融合

## 研究问题

原论文 EfficientNetV2M 采用冻结的 ImageNet 预训练主干和最后一层特征，在 7×7 特征图上用 valid 4×4 平均池化、Flatten 和分类头。复现实验中，原作者对输入像素重复缩放，导致三组数据的缺陷召回率均为 0%。前期独立的缩放与数据增强对照已表明，输入尺度纠正能够恢复该模型的识别能力；纠正属于实现修复，不作为本研究的结构创新。

改进实验面向细小局部缺陷与整体打印结构并存的情形。模型从三个不同分辨率的主干层同时获取信息，并学习每张图片、每个尺度的平均响应和峰值响应的融合比例。研究假设是这一结构有助于在保留整体形状信息的同时捕获局部缺陷特征；是否成立须由实验决定。

## 结构与消融

预训练 EfficientNetV2M 接受 224×224 RGB 浮点像素 0–255，仅使用 Keras 主干内置缩放。全程冻结主干（包括 BatchNorm），以下三个层的特征均在一次前向传播中获得：

| 层 | 形状 | 通道数 |
|---|---|---:|
| `block3e_add` | 28×28×80 | 80 |
| `block5n_add` | 14×14×176 | 176 |
| `top_activation` | 7×7×1280 | 1280 |

对每个尺度 i，将全局平均池化得到的向量 A_i 和全局最大池化得到的向量 M_i 分别做无训练参数的 LayerNormalization（epsilon=1e−5，center=False、scale=False）。对完整结构 F：

```
g_i = sigmoid(w_i · concat(A_i, M_i) + b_i)
f_i = A_i + g_i × (M_i − A_i)
f = concat(f_1, f_2, f_3)
p = Softmax(Dense_2(Dropout_0.5(ReLU(Dense_64(f)))))
```

三个门各自是一个神经元；初始化权重和偏置为零，因此初始时 g_i=0.5，F 与 E 的初始预测在相同随机种子下相同。F 中 g_i 随输入变化，由训练过程自动学习。融合向量维度 1536，输出是正常与缺陷两个类别的概率。其余层的参数初始化、分类头宽度和训练策略一致。

| 方案 | 实现 | 研究用途 |
|---|---|---|
| A | 原论文设置、原输入缩放 | 使用上一轮复现数据（1 种子）；背景参考 |
| B | 修正缩放、原 4×4 average pool/Flatten | 修正后的比较基线 |
| C | 深层 GAP＋固定归一化 | 完整深层特征对照（池化、归一化同时调整） |
| D | 三尺度 GAP＋固定归一化＋拼接 | 比 C 增加多尺度特征 |
| E | 三尺度 GAP/GMP 等权融合＋拼接 | 比 D 增加最大池化信息，权重固定 0.5 |
| F | 三尺度自适应双池化＋拼接 | 比 E 增加可学习门控（3075 个参数） |

## 实验协议

- 数据遵循 `reproduction/results_efficientnetv2m/{study}/split.csv`。按逐图清单读取，保证 B–F 的图像完全相同。Study One 271 张、Study Two 3017 张、Study Three 1557 张，共 4845 张图片。
- B–F 分别测试 Adam、SGD、RMSprop，学习率 0.001，用原来的 batch size/epoch：8/30、50/50、30/30。每次随机种子 42、43 或 44。相同 study/seed 的训练数据顺序一致，D/E/F 使用相同维度的分类头和初始化。损失函数为 binary crossentropy，采用最终轮权重，未增加数据增强或微调。135 次新训练，共计 144 组测试结果（含 9 组历史 A）。
- 验证集选择每一 study/variant 的优化器，以三种子缺陷 F1 平均值最大者为准，平局优先缺陷 recall，仍相同取固定顺序 Adam、SGD、RMSprop。提前记录选择，再计算全部预设对照的测试结果。A 使用历史模型，训练循环与本实验有差异，严格的结构消融使用 B–F。此前复现已使用过相同测试集，不能将其称为全新外部测试集。
- 使用固定列顺序：类别 0 为缺陷，1 为正常。缺陷为正类，argmax 分类（平局算缺陷）。accuracy、precision、recall、F1、specificity、AUROC、average precision 均从逐样本概率重新计算。均值与样本标准差按三个训练种子计算，反映随机初始化差异，不是置信区间；同样的测试图在不同模型中用逐图配对比较。
- SHA256 检查跨划分完全相同的文件。但 Study Two 为作者预先增强的图片，增强版本可能来自同一个原图，哈希不能检测所有增强变体。Study One/Three 也没有可靠的完整工件 ID，因此不对未见过工件的独立泛化做过度推断。
- 跨数据集：Study One 训练模型直接评估全部 Study Three 图片，反向同理。目标数据不用于本次源模型训练和优化器选择。排除重复文件仍不能保证所有图像来源独立；此实验只反映数据域变化下的表现，类别外观和拍摄设置可能不同。
- Grad-CAM 是定性、粗粒度的可解释性分析，显示每个分支对缺陷分类分数的梯度响应。无像素级定位标签，不能据此计算定位精度或断言高亮必然是正确缺陷区域。仅展示测试集中各类预测结果的第一个样本，防止只选成功案例。速度实测包括完整主干但不包括图片读取、缩放；各方案使用统一的特征提取与池化图，未专门最小化每个方案的部署计算图，属于本机近似推理性能。

## 重跑与验收

在 `reproduction/` 中执行（可继续中断的训练）：

```bash
.venv/bin/python innovation/run.py
.venv/bin/python innovation/report.py
.venv/bin/python innovation/explain.py
```

也可以使用 `--stage cache/train/evaluate` 和 `--study one/two/three` 单独分阶段执行；汇总与解释需要全部三组。已完成的训练通过配置和特征缓存身份检查后自动跳过。历史复现、原始数据和增强实验目录不被覆盖。

`results/` 输出：

- `summary.csv`：三组×六方案×三优化器的测试指标均值和标准差；A 只有一次历史复现，无标准差。
- `selected_summary.csv`：按验证 F1 选择优化器后的 B–F 总表；`all_test_runs.csv`：逐种子测试结果；`paired_ablation.csv`：相同 seed、优化器和测试图中，改动前后正确性变化。
- `cross_summary.csv`：跨数据集成绩；`gate_weights.csv`：F 的三个门控比例；`inference_benchmark.csv`：端到端延时及参数量。
- `ablation.svg`、`confusion_matrices.svg`、`gradcam_{one,two,three}.png`：可用于小组汇报的结果图。
- `*/B...F/优化器/种子/`：逐图验证和测试预测、训练日志、最终权重和完整运行配置。

本次实测及论文写作建议见 `results/结论与报告建议.md`。

相关文献及项目总体操作请见 [项目根目录 README](../README.md)。MSAF 是针对本任务的结构组合与受控实验，不宣称基础池化机制为全球首次提出。
