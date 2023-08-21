from PIL import Image
import os
import shutil
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', type=str)
    parser.add_argument('--target-dir', type=str)
    parser.add_argument('--target-resolution', type=int, default=256)
    args = parser.parse_args()

    target_resolution = (args.target_resolution, args.target_resolution)

    # 构建target目录结构
    for root, dirs, files in os.walk(args.source_dir):
        # 构建目标目录路径
        target_dir = os.path.join(args.target_dir, os.path.relpath(root, args.source_dir))
        # 创建目标目录（如果不存在）
        os.makedirs(target_dir, exist_ok=True)

    # # 遍历源数据集目录
    # for root, dirs, files in os.walk(args.source_dir):
    #     # 构建目标目录路径
    #     target_dir = os.path.join(target_dataset_dir, os.path.relpath(root, source_dataset_dir))
    #
    #     # 创建目标目录（如果不存在）
    #     os.makedirs(target_dir, exist_ok=True)
    #
    #     for file in files:
    #         source_file_path = os.path.join(root, file)
    #         target_file_path = os.path.join(target_dir, file)
    #
    #         # 如果是图片文件（以.jpg结尾），则压缩到目标分辨率
    #         if file.lower().endswith('.jpg'):
    #             with Image.open(source_file_path) as img:
    #                 img = img.resize(target_resolution, Image.ANTIALIAS)
    #                 img.save(target_file_path, "JPEG")
    #
    #         # 否则，直接复制文件
    #         else:
    #             shutil.copy2(source_file_path, target_file_path)
    #
    # print("压缩和复制完成")
    #
    # # 复制源数据集的目录结构到目标数据集
    # shutil.copytree(args.source_dir, args.target_dir)
    #
    # # 遍历目标数据集的文件夹和文件
    # for root, dirs, files in os.walk(args.target_dir):
    #     for filename in files:
    #         # 文件的完整路径
    #         file_path = os.path.join(root, filename)
    #
    #         # 如果文件是图像文件（以.jpg结尾），则调整分辨率并保存
    #         if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
    #             try:
    #                 with Image.open(file_path) as img:
    #                     img = img.resize(target_resolution, Image.ANTIALIAS)
    #                     img.save(file_path)
    #                 print(f"Resized {file_path}")
    #             except Exception as e:
    #                 print(f"Error processing {file_path}: {e}")
    #         else:
    #             # 如果文件不是图像文件，直接复制到目标目录
    #             relative_path = os.path.relpath(file_path, target_dataset_path)
    #             target_file_path = os.path.join(target_dataset_path, relative_path)
    #             os.makedirs(os.path.dirname(target_file_path), exist_ok=True)
    #             shutil.copy(file_path, target_file_path)
    #             print(f"Copied {file_path} to {target_file_path}")
