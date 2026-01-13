"""
SAM3 批量标注工具使用示例
"""

from batch_annotate_sam3 import BatchAnnotator

# 配置参数
MODEL_PATH = r"D:\qianpf\code\sam3-main\model\sam3\sam3.pt"  # 模型路径
IMAGE_FOLDER = r"C:\Users\29923\Desktop\3d_guajia_img\GA0SZT008103"  # 图片文件夹
OUTPUT_FOLDER = None  # None 表示保存到图片同目录，或指定输出文件夹路径

# 标注参数
TEXT_PROMPT = "rectangle"  # 提示词the front side of the 
SHAPE_TYPE = "polygon"  # "polygon" (分割) 或 "rectangle" (检测框)
CONF_THRESHOLD = 0.1 # 置信度阈值
MAX_OBJECTS = 1  # 每张图最大标注数量，None 表示不限制
LABEL = "guajia"  # 标签名称，None 表示使用提示词

# 创建标注器
annotator = BatchAnnotator(model_path=MODEL_PATH, device="cuda")

# 批量处理
annotator.process_folder(
    folder_path=IMAGE_FOLDER,
    text_prompt=TEXT_PROMPT,
    shape_type=SHAPE_TYPE,
    conf_threshold=CONF_THRESHOLD,
    max_objects=MAX_OBJECTS,
    label=LABEL,
    output_folder=OUTPUT_FOLDER
)

print("批量标注完成！")

