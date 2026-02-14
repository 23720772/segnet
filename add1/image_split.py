from PIL import Image
import matplotlib.pyplot as plt

# 加载图像
image_path = r"E:\messdior_dataset\Messidor_mine\IMAGES\20051020_45004_0100_PP.png"  # 替换为你的图片路径
img = Image.open(image_path)

# 定义行和列的数量
rows, cols = 3, 3

# 获取图像的宽度和高度
width, height = img.size

# 计算每张小图的宽度和高度
small_width = width // cols
small_height = height // rows

# 创建一个列表存储裁剪后的图像
cropped_images = []

# 将图像分割成9部分
for i in range(rows):
    for j in range(cols):
        left = j * small_width
        upper = i * small_height
        right = left + small_width
        lower = upper + small_height
        cropped_img = img.crop((left, upper, right, lower))
        cropped_images.append(cropped_img)

# 使用Matplotlib展示9张小图
fig, axes = plt.subplots(3, 3, figsize=(6, 6))

for idx, cropped_img in enumerate(cropped_images):
    ax = axes[idx // 3, idx % 3]
    ax.imshow(cropped_img)
    ax.axis('off')  # 关闭坐标轴
    ax.set_title(f"Image {idx + 1}")  # 设置编号

plt.tight_layout()
plt.show()
