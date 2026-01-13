# SAM3 批量标注工具使用说明

从 `sam3_labelme.py` 提取的批量标注功能，支持非交互式批量处理图片文件夹。

## 功能特点

- ✅ 批量处理文件夹中的所有图片
- ✅ 支持两种标注类型：`polygon` (分割) 或 `rectangle` (检测框)
- ✅ 自定义文本提示词
- ✅ 可设置置信度阈值
- ✅ 可限制每张图的标注数量
- ✅ 输出 labelme 格式的 JSON 文件

## 安装依赖

确保已安装以下依赖：
```bash
pip install torch numpy pillow scikit-image tqdm
```

## 使用方法

### 方法1: 命令行方式

```bash
python batch_annotate_sam3.py \
    --model_path "model/sam3/sam3.pt" \
    --folder "path/to/images" \
    --prompt "car" \
    --shape_type polygon \
    --conf_threshold 0.3 \
    --max_objects 10 \
    --label "car" \
    --output_folder "path/to/output" \
    --device cuda
```

#### 参数说明：

- `--model_path`: SAM3 模型路径（必需）
- `--folder`: 图片文件夹路径（必需）
- `--prompt`: 文本提示词，例如 "car", "person", "dog"（必需）
- `--shape_type`: 标注类型，`polygon` (分割) 或 `rectangle` (检测框)，默认 `polygon`
- `--conf_threshold`: 置信度阈值，默认 `0.3`
- `--max_objects`: 每张图最大标注数量，默认不限制
- `--label`: 标签名称，默认使用提示词
- `--output_folder`: 输出文件夹，默认保存到图片同目录
- `--device`: 设备，`cuda` 或 `cpu`，默认 `cuda`

### 方法2: Python 脚本方式

参考 `batch_annotate_example.py` 文件，修改配置参数后运行：

```python
from batch_annotate_sam3 import BatchAnnotator

# 创建标注器
annotator = BatchAnnotator(
    model_path="model/sam3/sam3.pt",
    device="cuda"
)

# 批量处理
annotator.process_folder(
    folder_path="path/to/images",
    text_prompt="car",
    shape_type="polygon",  # 或 "rectangle"
    conf_threshold=0.3,
    max_objects=10,
    label="car",
    output_folder=None  # None 表示保存到图片同目录
)
```

## 使用示例

### 示例1: 检测汽车（检测框模式）

```bash
python batch_annotate_sam3.py \
    --model_path "model/sam3/sam3.pt" \
    --folder "images" \
    --prompt "car" \
    --shape_type rectangle \
    --conf_threshold 0.3 \
    --max_objects 5 \
    --label "car"
```

### 示例2: 分割人物（多边形模式）

```bash
python batch_annotate_sam3.py \
    --model_path "model/sam3/sam3.pt" \
    --folder "images" \
    --prompt "person" \
    --shape_type polygon \
    --conf_threshold 0.5 \
    --max_objects 10 \
    --label "person"
```

### 示例3: 检测多个类别

如果需要检测多个类别，可以分别运行多次：

```bash
# 检测汽车
python batch_annotate_sam3.py --model_path "model/sam3/sam3.pt" --folder "images" --prompt "car" --shape_type rectangle --label "car"

# 检测人物
python batch_annotate_sam3.py --model_path "model/sam3/sam3.pt" --folder "images" --prompt "person" --shape_type rectangle --label "person"

# 检测狗
python batch_annotate_sam3.py --model_path "model/sam3/sam3.pt" --folder "images" --prompt "dog" --shape_type rectangle --label "dog"
```

## 输出格式

输出的 JSON 文件符合 labelme 格式，可以直接用 labelme 打开查看和编辑。

JSON 文件结构：
```json
{
  "version": "5.0.1",
  "flags": {},
  "shapes": [
    {
      "label": "car",
      "points": [[x1, y1], [x2, y2]],  // rectangle 模式
      // 或
      "points": [[x1, y1], [x2, y2], ...],  // polygon 模式
      "group_id": null,
      "shape_type": "rectangle",  // 或 "polygon"
      "flags": {}
    }
  ],
  "imagePath": "image.jpg",
  "imageData": null,
  "imageHeight": 480,
  "imageWidth": 640
}
```

## 注意事项

1. **模型路径**: 确保模型文件路径正确
2. **GPU 内存**: 如果 GPU 内存不足，可以使用 `--device cpu`（速度较慢）
3. **置信度阈值**: 根据实际效果调整，太低会检测到很多误检，太高会漏检
4. **标注数量限制**: `--max_objects` 会按置信度从高到低选择前 N 个结果
5. **输出位置**: 如果不指定 `--output_folder`，JSON 文件会保存到图片同目录

## 常见问题

**Q: 如何提高检测精度？**
A: 可以尝试：
- 提高置信度阈值 (`--conf_threshold`)
- 使用更具体的提示词
- 限制标注数量 (`--max_objects`) 只保留最置信的结果

**Q: 如何合并多个类别的标注？**
A: 可以分别运行多次，然后手动合并 JSON 文件，或者使用 labelme 打开多个 JSON 文件进行编辑。

**Q: 支持哪些图片格式？**
A: 支持 `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff` 格式。

**Q: 处理速度如何？**
A: 取决于图片数量和 GPU 性能。使用 GPU (`--device cuda`) 会快很多。

