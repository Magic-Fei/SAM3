# SAM3 批量推理工具使用指南

## 简介

`batch_inference.py` 是一个用于批量推理的脚本，可以对整个文件夹的图像进行自动推理，支持：
- ✅ 批量处理图像文件夹
- ✅ 保存可视化结果（overlays）
- ✅ 保存 mask 图像
- ✅ 转换为 labelme 格式（分割或检测）
- ✅ 支持 FP16 模型加速
- ✅ 支持多类别自动分类
- ✅ 多种类别分配策略

## 快速开始

### 1. 配置参数

编辑 `batch_inference.py` 文件顶部的配置区域：

```python
# 模型配置
CHECKPOINT_PATH = Path(r"D:\code\sam3-main\experiments_jiaodai\checkpoints\checkpoint_fp.pt")
INPUT_DIR = Path(r"D:\data\auxx\images")
OUTPUT_DIR = Path(r"D:\data\auxx\res")
PROMPT = "visual"
SCORE_THRESHOLD = 0.5
USE_FP16 = False

# 保存配置
SAVE_MASKS = False
SAVE_OVERLAYS = True
CONVERT_TO_LABELME = False
```

### 2. 运行推理

```bash
python tools/batch_inference.py
```

## 配置说明

### 基础配置

| 参数 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `CHECKPOINT_PATH` | Path | 训练好的模型 checkpoint 路径 | `Path(r"D:\model\checkpoint.pt")` |
| `INPUT_DIR` | Path | 输入图像文件夹路径 | `Path(r"D:\data\images")` |
| `OUTPUT_DIR` | Path | 输出结果文件夹路径 | `Path(r"D:\data\results")` |
| `PROMPT` | str | 文本提示词（通常使用 "visual"） | `"visual"` |
| `SCORE_THRESHOLD` | float | 置信度阈值，低于此值的检测结果会被过滤 | `0.5` |
| `USE_FP16` | bool | 是否使用 FP16 模型（需模型是 FP16 格式） | `False` |

### 保存配置

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `SAVE_MASKS` | bool | 是否保存单独的 mask 图像 | `False` |
| `SAVE_OVERLAYS` | bool | 是否保存可视化叠加图像 | `True` |
| `CONVERT_TO_LABELME` | bool | 是否转换为 labelme 格式 | `False` |

### Labelme 转换配置

#### 标注类型

```python
LABELME_ANNOTATION_TYPE = "segmentation"  # "segmentation" 或 "detection"
```

- **`segmentation`**: 分割模式，生成多边形标注
- **`detection`**: 检测模式，生成矩形框标注

#### 类别配置

**方式 1：多类别列表（推荐）**

```python
LABELME_CLASS_LABELS = ["guajia", "insulatingTube_fore", "jiaodai"]
```

**方式 2：单类别标签**

```python
LABELME_CLASS_LABEL = "object"  # 取消注释此方式时，注释掉上面的 LABELME_CLASS_LABELS
```

#### 类别分配策略

```python
LABELME_CLASS_ASSIGNMENT = "by_area"  # "round_robin"、"by_score" 或 "by_area"
```

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| `round_robin` | 循环分配：第1个=class1, 第2个=class2, 第3个=class3... | 检测结果按顺序对应不同类别 |
| `by_score` | 按置信度分配：根据分数阈值分配类别 | 不同类别有明确的置信度区分 |
| `by_area` | 按面积分组：面积相近的归为同一类别 | 不同类别在面积上有明显差异 |

#### 面积分组参数

```python
AREA_TOLERANCE = 0.8  # 面积容差（80%），面积相差在此范围内的归为同一类别
```

#### 置信度阈值（仅用于 `by_score` 模式）

```python
LABELME_CLASS_SCORE_THRESHOLDS = [0.5, 0.5, 0.5]  # 每个类别的置信度阈值
```

## 使用示例

### 示例 1：基础推理（仅可视化）

```python
CHECKPOINT_PATH = Path(r"D:\model\checkpoint.pt")
INPUT_DIR = Path(r"D:\data\images")
OUTPUT_DIR = Path(r"D:\data\results")
PROMPT = "visual"
SCORE_THRESHOLD = 0.5
USE_FP16 = False

SAVE_MASKS = False
SAVE_OVERLAYS = True
CONVERT_TO_LABELME = False
```

**输出：**
```
results/
└── overlays/
    ├── img001_overlay.png
    ├── img002_overlay.png
    └── ...
```

### 示例 2：保存分割标注（单类别）

```python
CHECKPOINT_PATH = Path(r"D:\model\checkpoint.pt")
INPUT_DIR = Path(r"D:\data\images")
OUTPUT_DIR = Path(r"D:\data\results")
PROMPT = "visual"
SCORE_THRESHOLD = 0.3

SAVE_OVERLAYS = True
CONVERT_TO_LABELME = True
LABELME_ANNOTATION_TYPE = "segmentation"
LABELME_CLASS_LABEL = "object"  # 单类别
```

**输出：**
```
results/
├── overlays/
│   └── ...
└── labelme_annotations/
    ├── img001.json
    ├── img002.json
    └── ...
```

### 示例 3：多类别分割（按面积分组）

```python
CHECKPOINT_PATH = Path(r"D:\model\checkpoint.pt")
INPUT_DIR = Path(r"D:\data\images")
OUTPUT_DIR = Path(r"D:\data\results")
PROMPT = "visual"
SCORE_THRESHOLD = 0.3

CONVERT_TO_LABELME = True
LABELME_ANNOTATION_TYPE = "segmentation"
LABELME_CLASS_LABELS = ["guajia", "insulatingTube_fore", "jiaodai"]
LABELME_CLASS_ASSIGNMENT = "by_area"
AREA_TOLERANCE = 0.8  # 80% 容差
```

**说明：**
- 检测结果按面积从大到小排序
- 面积相近的（相差在 80% 以内）归为同一类别
- 每组分配一个类别标签（循环使用）

### 示例 4：多类别检测（按置信度分配）

```python
CHECKPOINT_PATH = Path(r"D:\model\checkpoint.pt")
INPUT_DIR = Path(r"D:\data\images")
OUTPUT_DIR = Path(r"D:\data\results")
PROMPT = "visual"
SCORE_THRESHOLD = 0.2

CONVERT_TO_LABELME = True
LABELME_ANNOTATION_TYPE = "detection"
LABELME_CLASS_LABELS = ["class1", "class2", "class3"]
LABELME_CLASS_ASSIGNMENT = "by_score"
LABELME_CLASS_SCORE_THRESHOLDS = [0.8, 0.5, 0.3]
```

**说明：**
- 分数 >= 0.8 → class1
- 0.5 <= 分数 < 0.8 → class2
- 0.3 <= 分数 < 0.5 → class3
- 分数 < 0.3 → class3（最低类别）

### 示例 5：使用 FP16 模型加速

```python
CHECKPOINT_PATH = Path(r"D:\model\checkpoint_fp16.pt")
USE_FP16 = True  # 启用 FP16
DEVICE = torch.device("cuda")  # FP16 需要 CUDA
```

**注意：**
- FP16 模型文件更小（约 1GB vs 2GB）
- 推理速度更快
- 精度损失 <1%
- 仅支持 CUDA 设备

## 输出格式

### 1. 可视化图像（Overlays）

保存在 `{OUTPUT_DIR}/overlays/` 目录：
- 文件名格式：`{原图名}_overlay.png`
- 内容：原图 + 彩色 mask 叠加 + 边界框 + 置信度分数

### 2. Mask 图像

保存在 `{OUTPUT_DIR}/masks/` 目录（需 `SAVE_MASKS = True`）：
- 文件名格式：`{原图名}_mask_{序号:02d}.png`
- 内容：二值 mask 图像（白色=目标，黑色=背景）

### 3. Labelme 标注文件

保存在 `{OUTPUT_DIR}/labelme_annotations/` 目录（需 `CONVERT_TO_LABELME = True`）：

**分割格式（polygon）：**
```json
{
  "version": "5.8.3",
  "shapes": [
    {
      "label": "guajia",
      "points": [[x1, y1], [x2, y2], ...],
      "shape_type": "polygon",
      "flags": {}
    }
  ],
  "imagePath": "image.jpg",
  "imageData": null,
  "imageHeight": 1080,
  "imageWidth": 1920
}
```

**检测格式（rectangle）：**
```json
{
  "version": "5.8.3",
  "shapes": [
    {
      "label": "guajia",
      "points": [[x1, y1], [x2, y2]],
      "shape_type": "rectangle",
      "flags": {}
    }
  ],
  "imagePath": "image.jpg",
  "imageData": null,
  "imageHeight": 1080,
  "imageWidth": 1920
}
```

## 类别分配策略详解

### 1. Round Robin（循环分配）

```python
LABELME_CLASS_ASSIGNMENT = "round_robin"
LABELME_CLASS_LABELS = ["class1", "class2", "class3"]
```

**分配规则：**
- 第 1 个检测 → class1
- 第 2 个检测 → class2
- 第 3 个检测 → class3
- 第 4 个检测 → class1（循环）
- 第 5 个检测 → class2
- ...

**适用场景：** 检测结果按顺序对应不同类别

### 2. By Score（按置信度分配）

```python
LABELME_CLASS_ASSIGNMENT = "by_score"
LABELME_CLASS_LABELS = ["high_conf", "medium_conf", "low_conf"]
LABELME_CLASS_SCORE_THRESHOLDS = [0.8, 0.5, 0.3]
```

**分配规则：**
- 分数 >= 0.8 → high_conf
- 0.5 <= 分数 < 0.8 → medium_conf
- 0.3 <= 分数 < 0.5 → low_conf
- 分数 < 0.3 → low_conf（最低类别）

**适用场景：** 不同类别有明确的置信度区分

### 3. By Area（按面积分组）

```python
LABELME_CLASS_ASSIGNMENT = "by_area"
LABELME_CLASS_LABELS = ["large", "medium", "small"]
AREA_TOLERANCE = 0.8  # 80% 容差
```

**分配规则：**
1. 计算每个检测结果的面积
2. 按面积从大到小排序
3. 面积相近的（相差在 80% 以内）归为一组
4. 每组分配一个类别标签（循环使用）

**示例：**
```
检测结果面积: [10000, 9500, 9200, 5000, 4800, 1000, 900]

分组结果:
- 组1 (large): [10000, 9500, 9200]  # 面积相近，归为一组
- 组2 (medium): [5000, 4800]        # 面积相近，归为一组
- 组3 (small): [1000, 900]          # 面积相近，归为一组
```

**适用场景：** 不同类别在面积上有明显差异（如大、中、小目标）

## 常见问题

### Q1: 模型加载失败

**问题：** 提示 missing keys 或 unexpected keys

**解决方案：**
- 检查 checkpoint 格式是否正确
- 确保模型结构与 checkpoint 匹配
- 查看控制台输出的 missing/unexpected keys 信息

### Q2: 检测结果为空

**问题：** 没有检测到任何目标

**解决方案：**
- 降低 `SCORE_THRESHOLD`（如从 0.5 降到 0.3）
- 检查 `PROMPT` 是否正确（通常使用 "visual"）
- 确认模型是否针对目标类别训练过

### Q3: FP16 模型在 CPU 上运行失败

**问题：** 使用 FP16 模型时出错

**解决方案：**
- FP16 仅支持 CUDA 设备
- CPU 上会自动转换为 FP32
- 确保 `DEVICE = torch.device("cuda")`

### Q4: 类别分配不正确

**问题：** 多类别分配结果不符合预期

**解决方案：**
- 尝试不同的分配策略（`round_robin`、`by_score`、`by_area`）
- 调整 `AREA_TOLERANCE`（面积分组时）
- 调整 `LABELME_CLASS_SCORE_THRESHOLDS`（置信度分配时）
- 查看控制台输出的分组信息（调试用）

### Q5: 推理速度慢

**问题：** 处理大量图像时速度很慢

**解决方案：**
- 使用 GPU（`DEVICE = torch.device("cuda")`）
- 使用 FP16 模型（`USE_FP16 = True`）
- 提高 `SCORE_THRESHOLD` 减少后处理时间
- 关闭不需要的输出（`SAVE_MASKS = False`）

### Q6: Labelme 文件无法打开

**问题：** labelme 无法正确显示标注

**解决方案：**
- 确保 JSON 文件格式正确
- 检查 `imagePath` 是否指向正确的图像文件
- 使用 labelme 5.8.3 或更高版本
- 检查多边形点数是否 >= 3

## 工作流程建议

```
1. 准备模型和图像
   ├── 训练好的 checkpoint
   └── 待推理的图像文件夹

2. 配置参数
   ├── 设置路径（模型、输入、输出）
   ├── 设置推理参数（阈值、提示词）
   └── 设置输出选项（可视化、标注格式）

3. 运行推理
   python tools/batch_inference.py

4. 检查结果
   ├── 查看可视化图像（overlays）
   ├── 检查 labelme 标注文件
   └── 使用 labelme 打开验证

5. 调整参数（如需要）
   ├── 调整置信度阈值
   ├── 调整类别分配策略
   └── 重新运行推理
```

## 与训练流程集成

```bash
# 1. 训练模型
python sam3/train/train.py --config train_config.yaml

# 2. 压缩模型（可选）
python tools/compress_model.py

# 3. 批量推理
python tools/batch_inference.py

# 4. 使用 labelme 查看/编辑结果
labelme results/labelme_annotations/
```

## 提示和技巧

1. **置信度阈值选择**
   - 精确优先：0.5-0.7
   - 平衡：0.3-0.5
   - 召回优先：0.1-0.3

2. **面积分组容差**
   - 严格分组：0.3-0.5（30%-50% 容差）
   - 中等分组：0.5-0.8（50%-80% 容差）
   - 宽松分组：0.8-1.0（80%-100% 容差）

3. **多类别配置**
   - 类别数量应与训练时的类别数一致
   - 类别名称建议使用有意义的名称
   - 可以先用单类别测试，再扩展到多类别

4. **性能优化**
   - 使用 FP16 模型可节省约 50% 显存
   - 批量处理时关闭不必要的输出
   - 使用 SSD 存储可加快 I/O 速度

---

**开始使用：**
```bash
# 1. 编辑配置
vim tools/batch_inference.py

# 2. 运行推理
python tools/batch_inference.py
```

祝推理顺利！ 🚀

