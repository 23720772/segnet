import os
import random
from PIL import Image, ImageDraw, ImageFont

# 定义输入文件夹路径
image_folder_path = r'S:\数据集\Defects location for metal surface\Augmented\YOLO_Images_Erased'
label_folder_path = r'S:\数据集\Defects location for metal surface\Augmented\YOLO_Annotations_x_Erased'
output_folder_path = r'S:\数据集\Defects location for metal surface\Augmented\Annotated_Images'

# 确保输出文件夹存在
os.makedirs(output_folder_path, exist_ok=True)

# 获取所有图片文件名
all_images = [f for f in os.listdir(image_folder_path) if f.endswith('.jpg')]

# 设置需要标注的图片数量
num_annotate = 500

# 打乱图片文件名列表并选择前num_annotate个
random.shuffle(all_images)
annotate_images = all_images[:num_annotate]

# 定义类别ID到类别名称的映射
id_to_class_name = ['1_chongkong', '2_hanfeng', '3_yueyawan', '4_shuiban', '5_youban', '6_siban', '7_yiwu', '8_yahen',
                    '9_zhehen', '10_yaozhe']

# 定义字体大小
font_size = 36
font = ImageFont.truetype("arial.ttf", font_size)


def draw_annotations(image_path, label_path, output_path):
    # 打开图片
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # 读取标签文件
    with open(label_path, 'r') as f:
        labels = f.readlines()

    # 获取图片尺寸
    w, h = image.size

    # 解析标签并绘制矩形框
    for label in labels:
        parts = label.strip().split()
        class_id = int(parts[0])
        bbox = list(map(float, parts[1:]))

        # 转换bbox到图片坐标
        x_center, y_center, width, height = bbox
        x_min = (x_center - width / 2) * w
        y_min = (y_center - height / 2) * h
        x_max = (x_center + width / 2) * w
        y_max = (y_center + height / 2) * h

        # 绘制矩形框
        draw.rectangle([x_min, y_min, x_max, y_max], outline='white', width=2)

        # 确定文本位置
        text_position = (x_min, y_min - font_size if y_min - font_size > 0 else y_min)
        # 绘制文本
        draw.text(text_position, f"{id_to_class_name[class_id]} ({class_id})", fill='white', font=font)

    # 保存标注后的图片
    image.save(output_path)


# 标注指定数量的图片
for image_name in annotate_images:
    base_name = os.path.splitext(image_name)[0]
    label_name = base_name + '.txt'

    image_path = os.path.join(image_folder_path, image_name)
    label_path = os.path.join(label_folder_path, label_name)
    output_path = os.path.join(output_folder_path, image_name)

    if os.path.exists(image_path) and os.path.exists(label_path):
        draw_annotations(image_path, label_path, output_path)

print("标注完成！")
