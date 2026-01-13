# SAM3 标注工具完整使用指南

## 目录
1. [快速开始](#快速开始)
2. [GUI 标注工具](#gui-标注工具)
3. [批量标注脚本](#批量标注脚本)
4. [标注格式说明](#标注格式说明)
5. [高级用法](#高级用法)
6. [常见问题](#常见问题)

---

## 快速开始

### 1. 安装依赖

```bash
# 安装主项目依赖
pip install -r requirements.txt

# 安装标注工具额外依赖
pip install PyQt5 scikit-image
```

### 2. 准备模型

确保你有 SAM3 模型文件，推荐路径：
```
experiments/checkpoints/model_fp16.pt
```

### 3. 启动标注工具

**Windows:**
```bash
start_annotator.bat
```

**Linux/Mac:**
```bash
bash start_annotator.sh
```

或直接运行：
```bash
python sam3_annotator.py
```

---

## GUI 标注工具

### 界面布局

```
┌──────────────────────────────────────────────────────────┐
│  SAM3 Annotation Tool                                    │
├──────────┬──────────────────────────────┬────────────────┤
│          │                              │                │
│ 模型设置 │                              │   标注列表     │
│          │                              │                │
│ 图像操作 │        画布区域              │   [标注1]      │
│          │                              │   [标注2]      │
│ 标注模式 │      (显示图像和标注)        │   [标注3]      │
│          │                              │                │
│ 标注参数 │                              │   [删除]       │
│          │                              │   [清空]       │
│          │                              │   [保存JSON]   │
└──────────┴──────────────────────────────┴────────────────┘
```

### 工作流程

#### 第一步：加载模型

1. 点击 **"浏览模型"** 按钮
2. 选择模型文件（如 `experiments/checkpoints/model_fp16.pt`）
3. 点击 **"加载模型"** 按钮
4. 等待状态变为绿色 **"模型: 已加载"**

#### 第二步：加载图像

1. 点击 **"加载图像"** 按钮
2. 选择要标注的图像文件
3. 图像将显示在中央画布区域

#### 第三步：选择标注模式

##### 模式 A：文本提示（推荐用于批量标注）

**适用场景：** 需要检测图像中所有某类目标

**操作步骤：**
1. 选择 **"文本提示"** 模式
2. 在 **"文本提示"** 输入框输入类别，例如：
   - `car` - 检测所有汽车
   - `person` - 检测所有人
   - `dog` - 检测所有狗
3. 在 **"当前标签"** 输入框输入标签名称（可与提示词相同）
4. 调整 **"置信度阈值"**（推荐 0.3-0.5）
5. 点击 **"运行文本提示"**

**示例：**
```
文本提示: car
当前标签: car
置信度阈值: 0.3

结果: 自动检测图像中所有汽车
```

##### 模式 B：点提示（推荐用于精确标注）

**适用场景：** 需要精确标注单个复杂目标

**操作步骤：**
1. 选择 **"点提示"** 模式
2. 在图像上添加点：
   - **左键点击** → 前景点（目标内部）
   - **右键点击** → 背景点（不要的区域）
3. 通常添加 3-5 个点即可
4. 点击 **"运行点提示"**
5. 点击 **"清除点"** 可以重新开始

**示例：**
```
任务: 标注一只被遮挡的猫

操作:
1. 左键点击猫的头部（前景）
2. 左键点击猫的身体（前景）
3. 右键点击旁边的狗（背景，不要包含）
4. 运行点提示

结果: 精确分割出这只猫，不包含狗
```

##### 模式 C：框提示（推荐用于快速标注）

**适用场景：** 快速标注规则形状的目标

**操作步骤：**
1. 选择 **"框提示"** 模式
2. 在图像上按住鼠标左键拖拽
3. 绘制包含目标的矩形框
4. 释放鼠标，自动生成 mask

**示例：**
```
任务: 标注一辆车

操作:
1. 在车的左上角按下鼠标
2. 拖拽到右下角
3. 释放鼠标

结果: 自动精确分割出车辆
```

#### 第四步：编辑标注

- **修改标签：** 双击右侧列表中的标注项，输入新标签
- **删除标注：** 选中标注，点击 **"删除"** 按钮
- **清空所有：** 点击 **"清空"** 按钮

#### 第五步：保存标注

1. 点击 **"保存为 JSON"** 按钮
2. 选择保存位置（默认与图像同目录同名）
3. 生成 labelme 格式的 JSON 文件

---

## 批量标注脚本

### 基本用法

```bash
python batch_annotate.py \
    --image_dir ./images \
    --output_dir ./annotations \
    --model_path ./experiments/checkpoints/model_fp16.pt \
    --prompts car person
```

### 完整参数说明

| 参数 | 必需 | 说明 | 默认值 |
|------|------|------|--------|
| `--image_dir` | ✓ | 输入图像目录 | - |
| `--output_dir` | ✓ | 输出 JSON 目录 | - |
| `--model_path` | ✓ | SAM3 模型路径 | - |
| `--prompts` | ✓ | 类别名称列表 | - |
| `--threshold` | ✗ | 置信度阈值 | 0.3 |
| `--no_image_data` | ✗ | 不保存图像数据 | False |
| `--extensions` | ✗ | 图像扩展名 | .jpg .jpeg .png .bmp |

### 使用示例

#### 示例 1：标注单个类别

```bash
python batch_annotate.py \
    --image_dir ./street_images \
    --output_dir ./street_annotations \
    --model_path ./experiments/checkpoints/model_fp16.pt \
    --prompts car
```

#### 示例 2：标注多个类别

```bash
python batch_annotate.py \
    --image_dir ./coco_images \
    --output_dir ./coco_annotations \
    --model_path ./experiments/checkpoints/model_fp16.pt \
    --prompts person car bicycle motorcycle bus truck traffic_light \
    --threshold 0.4
```

#### 示例 3：不保存图像数据（减小文件大小）

```bash
python batch_annotate.py \
    --image_dir ./large_dataset \
    --output_dir ./annotations \
    --model_path ./experiments/checkpoints/model_fp16.pt \
    --prompts car \
    --no_image_data
```

### 批量标注工作流程

```
1. 准备数据
   ./images/
   ├── img001.jpg
   ├── img002.jpg
   └── img003.jpg

2. 运行批量标注
   python batch_annotate.py --image_dir ./images --output_dir ./annotations \
       --model_path ./model.pt --prompts car person

3. 生成标注文件
   ./annotations/
   ├── img001.json
   ├── img002.json
   └── img003.json

4. 使用 labelme 查看/编辑
   labelme ./annotations/
```

---

## 标注格式说明

### Labelme JSON 格式

生成的 JSON 文件与 labelme 完全兼容：

```json
{
  "version": "5.0.1",
  "flags": {},
  "shapes": [
    {
      "label": "car",
      "points": [
        [123.5, 234.6],
        [125.7, 236.8],
        ...
      ],
      "group_id": null,
      "shape_type": "polygon",
      "flags": {}
    }
  ],
  "imagePath": "image.jpg",
  "imageData": "base64_encoded_string_or_null",
  "imageHeight": 1080,
  "imageWidth": 1920
}
```

### 字段说明

- **version**: labelme 版本号
- **shapes**: 标注对象列表
  - **label**: 类别标签
  - **points**: 多边形顶点坐标
  - **shape_type**: 形状类型（polygon 或 rectangle）
- **imagePath**: 图像文件名
- **imageData**: base64 编码的图像数据（可选）
- **imageHeight/imageWidth**: 图像尺寸

---

## 高级用法

### 1. 组合使用多种标注模式

```
场景: 标注一张包含多个目标的复杂图像

步骤:
1. 使用文本提示检测所有汽车
2. 使用点提示精确标注被遮挡的行人
3. 使用框提示快速标注交通标志
4. 统一保存为一个 JSON 文件
```

### 2. 自定义标签颜色

编辑 `annotator_config.yaml`:

```yaml
label_colors:
  my_custom_class: [255, 100, 50]  # RGB 颜色
```

### 3. 程序化标注

```python
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from PIL import Image

# 加载模型
model = build_sam3_image_model(
    checkpoint_path="./model.pt",
    load_from_HF=False
)
processor = Sam3Processor(model)

# 加载图像
image = Image.open("image.jpg")
state = processor.set_image(image)

# 文本提示
output = processor.set_text_prompt(state=state, prompt="car")
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

# 处理结果...
```

### 4. 与训练流程集成

```bash
# 1. 标注数据
python batch_annotate.py --image_dir ./raw_images --output_dir ./annotations \
    --model_path ./model.pt --prompts target_class

# 2. 转换格式（如需要）
python convert_labelme_to_coco.py --input ./annotations --output ./coco_format

# 3. 训练模型
python sam3/train/train.py --config ./train_config.yaml
```

---

## 常见问题

### Q1: 模型加载失败

**问题：** 点击"加载模型"后出错

**解决方案：**
- 检查模型文件路径是否正确
- 确保模型文件完整（大小约 170MB）
- 检查 PyTorch 版本是否兼容

### Q2: 检测结果不准确

**问题：** 文本提示检测到错误的目标

**解决方案：**
- 使用更具体的类别名称（如 "sedan car" 而不是 "car"）
- 调整置信度阈值（提高以减少误检）
- 改用点提示或框提示模式手动标注

### Q3: 标注边界不准确

**问题：** 生成的 mask 边界不够精确

**解决方案：**
- 使用点提示模式，添加更多前景点和背景点
- 在目标边缘添加点以获得更精确的边界
- 调整 `annotator_config.yaml` 中的 `polygon_tolerance` 参数

### Q4: JSON 文件过大

**问题：** 生成的 JSON 文件占用大量存储空间

**解决方案：**
- 使用 `--no_image_data` 参数不保存 base64 图像数据
- 增加 `polygon_tolerance` 简化多边形
- 使用更高的置信度阈值减少标注数量

### Q5: 批量标注速度慢

**问题：** 处理大量图像时速度很慢

**解决方案：**
- 使用 GPU 加速（自动检测）
- 减少 prompts 数量
- 提高置信度阈值
- 使用更小的模型（如果可用）

### Q6: 如何编辑已有标注

**问题：** 需要修改之前保存的标注

**解决方案：**
```bash
# 方法1: 使用 GUI 工具
python sam3_annotator.py
# 点击"加载 JSON"，修改后重新保存

# 方法2: 使用 labelme
labelme annotation.json

# 方法3: 直接编辑 JSON 文件
```

### Q7: 与其他标注格式转换

**问题：** 需要 COCO、YOLO 等格式

**解决方案：**
```bash
# Labelme 自带转换工具
labelme_json_to_dataset annotation.json

# 或使用第三方转换脚本
# https://github.com/wkentaro/labelme
```

---

## 技巧和最佳实践

### 标注技巧

1. **文本提示选择**
   - ✓ 好: "car", "person", "dog"
   - ✗ 差: "vehicles", "things", "stuff"

2. **点提示策略**
   - 先添加目标中心的前景点
   - 在目标边缘添加更多前景点
   - 在邻近干扰物上添加背景点

3. **置信度阈值**
   - 精确标注: 0.4-0.6
   - 召回优先: 0.2-0.3
   - 准确优先: 0.5-0.7

### 工作流程建议

```
数据准备 → 批量自动标注 → GUI 工具微调 → 质量检查 → 导出使用
   ↓            ↓              ↓            ↓          ↓
 整理图像   batch_annotate  sam3_annotator  labelme   训练模型
```

### 质量控制

1. **随机抽查**：定期检查标注质量
2. **边界检查**：确保 mask 边界准确
3. **类别一致**：保持标签命名统一
4. **覆盖完整**：确保所有目标都被标注

---

## 总结

SAM3 标注工具提供了：

- 🎨 **直观的 GUI 界面** - 类似 labelme 的操作体验
- 🤖 **AI 辅助标注** - 大幅提升标注效率
- 📦 **批量处理能力** - 支持大规模数据标注
- 🔄 **格式兼容性** - 完全兼容 labelme 格式
- ✏️ **灵活的编辑** - 支持标注微调

**开始使用：**
```bash
python sam3_annotator.py
```

**批量标注：**
```bash
python batch_annotate.py --image_dir ./images --output_dir ./annotations \
    --model_path ./model.pt --prompts car person
```

祝标注愉快！ 🚀

