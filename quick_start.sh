#!/bin/bash
# SAM3 单类别分割训练快速开始脚本（硬编码版本）

set -e

# ============================================================================
# 硬编码配置 - 请修改以下参数
# ============================================================================
# 注意: 这些配置需要与 labelme_to_coco.py 中的配置保持一致
LABELME_DIR="C:/Users/Fei/Desktop/data/test"      # Labelme 标注文件目录
OUTPUT_DIR="C:/Users/Fei/Desktop/data/coco"       # 输出 COCO 数据集目录
CLASS_NAME="guajia"                                # 类别名称
TRAIN_SPLIT=0.8                                    # 训练集比例 (0.8 = 80% 训练, 20% 验证)
EXPERIMENT_DIR="C:/Users/Fei/Desktop/experiments"  # 实验输出目录
# ============================================================================

echo "=========================================="
echo "SAM3 单类别分割训练快速开始"
echo "=========================================="
echo ""
echo "配置:"
echo "  Labelme 目录: $LABELME_DIR"
echo "  输出目录:     $OUTPUT_DIR"
echo "  类别名称:     $CLASS_NAME"
echo "  训练集比例:   $TRAIN_SPLIT"
echo "  实验目录:     $EXPERIMENT_DIR"
echo ""

# 检查 Labelme 目录
if [ ! -d "$LABELME_DIR" ]; then
    echo "错误: Labelme 目录不存在: $LABELME_DIR"
    echo "请修改脚本中的 LABELME_DIR 变量"
    exit 1
fi

# 检查是否有 JSON 文件
JSON_COUNT=$(find "$LABELME_DIR" -maxdepth 1 -name "*.json" 2>/dev/null | wc -l)
if [ $JSON_COUNT -eq 0 ]; then
    echo "错误: 在 $LABELME_DIR 中未找到 JSON 文件"
    exit 1
fi

echo "找到 $JSON_COUNT 个标注文件"
echo ""

# 步骤 1: 转换数据集
echo "步骤 1: 转换数据集格式..."
echo "注意: 请确保 labelme_to_coco.py 中的配置与脚本中的配置一致"
python labelme_to_coco.py

if [ $? -ne 0 ]; then
    echo "错误: 数据集转换失败"
    echo "请检查 labelme_to_coco.py 中的配置是否正确"
    exit 1
fi

echo ""
echo "数据集转换完成！"
echo ""

# 步骤 2: 提示修改配置文件
echo "步骤 2: 请修改 train_config_single_class.yaml 中的以下路径:"
echo ""
echo "  paths:"
echo "    dataset_root: $OUTPUT_DIR"
echo "    experiment_log_dir: $EXPERIMENT_DIR"
echo ""
echo "  dataset:"
echo "    class_name: \"$CLASS_NAME\""
echo ""

# 步骤 3: 提示开始训练
echo "步骤 3: 修改配置文件后，运行以下命令开始训练:"
echo ""
echo "  python sam3/train/train.py -c train_config_single_class.yaml"
echo ""

echo "=========================================="
echo "完成！"
echo "=========================================="

