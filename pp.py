import os
import shutil

def remove_spaces_in_filenames(folder_path):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件是否为图片文件（这里假设只考虑常见的图片格式）
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
            # 移除文件名中的空格
            new_filename = filename.replace(" ", "")
            # 构建文件的完整路径
            old_filepath = os.path.join(folder_path, filename)
            new_filepath = os.path.join(folder_path, new_filename)
            # 重命名文件
            os.rename(old_filepath, new_filepath)

# 指定文件夹路径
folder_path = "inputs/val/masks/0"

# 调用函数去除文件名中的空格
remove_spaces_in_filenames(folder_path)



# import os
# import shutil
#
# def remove_mask_from_filenames(folder_path):
#     # 遍历文件夹中的所有文件
#     for filename in os.listdir(folder_path):
#         # 检查文件是否为图片文件（这里假设只考虑常见的图片格式）
#         if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
#             # 移除文件名中的"_mask"
#             new_filename = filename.replace("_mask", "")
#             # 构建文件的完整路径
#             old_filepath = os.path.join(folder_path, filename)
#             new_filepath = os.path.join(folder_path, new_filename)
#             # 重命名文件
#             os.rename(old_filepath, new_filepath)
#
# # 指定文件夹路径
# folder_path = "inputs/dsb2018_96/masks/0"
#
# # 调用函数移除文件名中的"_mask"
# remove_mask_from_filenames(folder_path)
