import torch ,os
import matplotlib.pyplot as plt
#################################### For Image ####################################
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import plot_results
import numpy as np
# 方式1: 指定本地模型文件路径（推荐）
# 如果你已经下载了模型文件到本地，可以直接指定路径
model = build_sam3_image_model(
    checkpoint_path=r""D:\qianpf\code\sam3-main\experiments\checkpoints\checkpoint_4.pt"",  # 指定本地模型文件路径
    load_from_HF=False  # 设置为 False 避免从 HuggingFace 下载
)

# 方式2: 从 HuggingFace 下载（默认行为）
# 模型会下载到 HuggingFace 缓存目录（通常是 ~/.cache/huggingface/hub/）
# 如果想指定 HuggingFace 缓存目录，可以设置环境变量：
# import os
# os.environ['HF_HOME'] = r'D:\path\to\huggingface_cache'  # 设置缓存目录

processor = Sam3Processor(model)
# Load an image
image = Image.open(r"C:\Users\29923\Desktop\1\2.jpeg")
inference_state = processor.set_image(image)
# Prompt the model with text
output = processor.set_text_prompt(state=inference_state, prompt="rectangles")

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]


# 打印结果信
print(f"检测到 {len(scores)} 个对象")
for i, (box, score) in enumerate(zip(boxes, scores)):
    print(f"对象 {i+1}: 置信度={score.item():.4f}, 边界框={box.cpu().tolist()}")

# 可视化结果
results = {
    "masks": masks,
    "boxes": boxes,
    "scores": scores
}
plot_results(image, results)
plt.title(f"SAM3 检测结果 - 提示词: 'box'", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()