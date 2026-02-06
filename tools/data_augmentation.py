import os
import random
import json
from PIL import Image, ImageOps, ImageFilter


def update_points_after_flip(points, image_width, image_height, flip_horizontal, flip_vertical, shape_type='polygon'):
    """
    根据翻转情况更新标注点坐标
    
    Args:
        points: 标注点列表
        image_width: 图片宽度
        image_height: 图片高度
        flip_horizontal: 是否水平翻转
        flip_vertical: 是否垂直翻转
        shape_type: 形状类型 ('rectangle' 或 'polygon')
    
    Returns:
        更新后的点列表
    """
    updated_points = []
    for x, y in points:
        new_x, new_y = x, y
        # 水平翻转：x坐标关于中心对称
        if flip_horizontal:
            new_x = image_width - x
        # 垂直翻转：y坐标关于中心对称
        if flip_vertical:
            new_y = image_height - y
        updated_points.append([new_x, new_y])
    
    # 对于rectangle类型，确保点的顺序正确（左上角和右下角）
    if shape_type == 'rectangle' and len(updated_points) == 2:
        x1, y1 = updated_points[0]
        x2, y2 = updated_points[1]
        # 确保第一个点是左上角，第二个点是右下角
        min_x, max_x = min(x1, x2), max(x1, x2)
        min_y, max_y = min(y1, y2), max(y1, y2)
        updated_points = [[min_x, min_y], [max_x, max_y]]
    
    return updated_points


def augment_image(image, augment_type='random'):
    """
    对图片进行数据增广
    
    Args:
        image: PIL Image对象
        augment_type: 增广类型
            - 'random': 随机组合翻转、曝光、模糊
            - 'flip': 仅翻转
            - 'exposure': 仅曝光调整
            - 'blur': 仅模糊
            - 'flip_exposure': 翻转+曝光
            - 'all': 所有增广方式
    
    Returns:
        处理后的图片和增广信息字典
    """
    aug_info = {
        'flip_horizontal': False,
        'flip_vertical': False,
        'exposure_factor': 1.0,
        'blur_radius': 0
    }
    
    processed_image = image.copy()
    
    if augment_type == 'random':
        # 随机水平翻转 (80%概率)
        if random.random() < 0.8:
            processed_image = ImageOps.mirror(processed_image)
            aug_info['flip_horizontal'] = True
        
        # 随机垂直翻转 (80%概率)
        if random.random() < 0.8:
            processed_image = ImageOps.flip(processed_image)
            aug_info['flip_vertical'] = True
        
        # 随机曝光调整 (70%概率)
        if random.random() < 0.7:
            exposure_factor = random.uniform(0.5, 2.0)
            processed_image = processed_image.point(lambda p: p * exposure_factor)
            aug_info['exposure_factor'] = exposure_factor
        
        # 随机高斯模糊 (50%概率)
        if random.random() < 0.2:
            blur_radius = random.uniform(1, 3)
            processed_image = processed_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            aug_info['blur_radius'] = blur_radius
    
    elif augment_type == 'flip':
        if random.random() < 0.5:
            processed_image = ImageOps.mirror(processed_image)
            aug_info['flip_horizontal'] = True
        if random.random() < 0.5:
            processed_image = ImageOps.flip(processed_image)
            aug_info['flip_vertical'] = True
    
    elif augment_type == 'exposure':
        exposure_factor = random.uniform(0.5, 2.0)
        processed_image = processed_image.point(lambda p: p * exposure_factor)
        aug_info['exposure_factor'] = exposure_factor
    
    elif augment_type == 'blur':
        blur_radius = random.uniform(1, 3)
        processed_image = processed_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        aug_info['blur_radius'] = blur_radius
    
    elif augment_type == 'flip_exposure':
        if random.random() < 0.5:
            processed_image = ImageOps.mirror(processed_image)
            aug_info['flip_horizontal'] = True
        if random.random() < 0.5:
            processed_image = ImageOps.flip(processed_image)
            aug_info['flip_vertical'] = True
        exposure_factor = random.uniform(0.5, 2.0)
        processed_image = processed_image.point(lambda p: p * exposure_factor)
        aug_info['exposure_factor'] = exposure_factor
    
    elif augment_type == 'all':
        # 水平翻转
        if random.random() < 0.5:
            processed_image = ImageOps.mirror(processed_image)
            aug_info['flip_horizontal'] = True
        # 垂直翻转
        if random.random() < 0.5:
            processed_image = ImageOps.flip(processed_image)
            aug_info['flip_vertical'] = True
        # 曝光调整
        exposure_factor = random.uniform(0.5, 2.0)
        processed_image = processed_image.point(lambda p: p * exposure_factor)
        aug_info['exposure_factor'] = exposure_factor
        # 高斯模糊
        blur_radius = random.uniform(1, 3)
        processed_image = processed_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        aug_info['blur_radius'] = blur_radius
    
    return processed_image, aug_info


def augment_images_with_json(
    input_folder, 
    output_folder, 
    augment_per_image=5,
    generate_json=True,
    augment_type='random'
):
    """
    对图片进行数据增广，可选择是否生成对应的JSON文件
    
    支持LabelMe格式的JSON标注文件，可以同时处理：
    - rectangle类型（目标检测，2个点：左上角和右下角）
    - polygon类型（分割，多个点：多边形顶点）
    
    Args:
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径
        augment_per_image: 每张图片增广的数量
        generate_json: 是否生成对应的JSON文件（如果存在原始JSON）
        augment_type: 增广类型 ('random', 'flip', 'exposure', 'blur', 'flip_exposure', 'all')
    """
    # 检查输出文件夹是否存在，如果不存在则创建
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取所有图片文件
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    image_files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith(image_extensions)
    ]
    
    if not image_files:
        print(f"在 {input_folder} 中未找到图片文件")
        return
    
    print(f"找到 {len(image_files)} 张图片，每张将增广 {augment_per_image} 次")
    
    # 遍历所有图片文件
    for image_filename in image_files:
        # 分离文件名和扩展名
        image_name, image_ext = os.path.splitext(image_filename)
        # 构造对应的json文件名
        json_filename = f"{image_name}.json"
        
        # 构建输入路径
        input_image_path = os.path.join(input_folder, image_filename)
        input_json_path = os.path.join(input_folder, json_filename)
        
        # 检查是否存在JSON文件
        has_json = os.path.exists(input_json_path) and generate_json
        
        try:
            # 打开原始图片
            with Image.open(input_image_path) as original_image:
                image_width, image_height = original_image.size
                
                # 读取JSON数据（如果存在且需要生成）
                json_data = None
                if has_json:
                    with open(input_json_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                
                # 对每张图片进行指定次数的增广
                for aug_idx in range(augment_per_image):
                    # 进行数据增广
                    augmented_image, aug_info = augment_image(original_image, augment_type)
                    
                    # 生成输出文件名
                    output_image_filename = f"{image_name}_aug{aug_idx+1}{image_ext}"
                    output_image_path = os.path.join(output_folder, output_image_filename)
                    
                    # 保存增广后的图片
                    augmented_image.save(output_image_path)
                    print(f"已保存: {output_image_filename} (翻转H:{aug_info['flip_horizontal']}, "
                          f"V:{aug_info['flip_vertical']}, 曝光:{aug_info['exposure_factor']:.2f}, "
                          f"模糊:{aug_info['blur_radius']:.2f})")
                    
                    # 处理对应的JSON文件（如果存在且需要生成）
                    if has_json and json_data is not None:
                        # 创建JSON数据的副本
                        new_json_data = json.loads(json.dumps(json_data))
                        
                        # 更新JSON中的图像路径
                        new_json_data['imagePath'] = output_image_filename
                        
                        # 更新标注点坐标（如果有翻转）
                        if aug_info['flip_horizontal'] or aug_info['flip_vertical']:
                            for shape in new_json_data.get('shapes', []):
                                if 'points' in shape:
                                    # 获取形状类型，默认为polygon
                                    shape_type = shape.get('shape_type', 'polygon')
                                    shape['points'] = update_points_after_flip(
                                        shape['points'], 
                                        image_width, 
                                        image_height, 
                                        aug_info['flip_horizontal'], 
                                        aug_info['flip_vertical'],
                                        shape_type=shape_type
                                    )
                        
                        # 保存更新后的JSON文件
                        output_json_filename = f"{image_name}_aug{aug_idx+1}.json"
                        output_json_path = os.path.join(output_folder, output_json_filename)
                        with open(output_json_path, 'w', encoding='utf-8') as f:
                            json.dump(new_json_data, f, ensure_ascii=False, indent=2)
                        print(f"  已保存JSON: {output_json_filename}")
                
        except Exception as e:
            print(f"处理 {image_filename} 时出错: {str(e)}")
    
    print(f"\n数据增广完成！共处理 {len(image_files)} 张图片，生成 {len(image_files) * augment_per_image} 张增广图片")


if __name__ == "__main__":
    # 配置参数
    input_folder = r'D:\qianpf\data\auxx\test'  # 输入文件夹路径
    output_folder = r'D:\qianpf\data\auxx\test_aug'  # 输出文件夹路径
    augment_per_image = 5 # 每张图片增广的数量
    generate_json = True  # 是否生成对应的JSON文件（如果存在原始JSON）
    augment_type = 'exposure'  # 增广类型: 'random', 'flip', 'exposure', 'blur', 'flip_exposure', 'all'
    
    # 执行数据增广
    augment_images_with_json(
        input_folder=input_folder,
        output_folder=output_folder,
        augment_per_image=augment_per_image,
        generate_json=generate_json,
        augment_type=augment_type
    )

