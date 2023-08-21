from PIL import Image
import os
import argparse
import shutil


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', type=str)
    parser.add_argument('--target-dir', type=str)
    parser.add_argument('--target-resolution', type=int, default=256)
    args = parser.parse_args()

    target_resolution = (args.target_resolution, args.target_resolution)

    # 遍历所有子目录
    total = len(os.listdir(args.source_dir))
    idx = 0
    for dirname in sorted(os.listdir(args.source_dir)):
        target_dir_path = os.path.join(args.target_dir, dirname)
        current_dir_path = os.path.join(args.source_dir, dirname)
        if dirname.endswith('.json'):
            shutil.copy(current_dir_path, target_dir_path)
            break
        # 创建文件夹
        os.makedirs(target_dir_path, exist_ok=True)
        # 拷贝文件
        for filename in sorted(os.listdir(current_dir_path)):
            current_file_path = os.path.join(current_dir_path, filename)
            target_file_path = os.path.join(target_dir_path, filename)
            # 压缩图片并拷贝
            if filename.endswith('.jpg'):
                with Image.open(current_file_path) as img:
                    img = img.resize(target_resolution, Image.ANTIALIAS)
                    img.save(target_file_path, "JPEG")
            # 直接复制其他文件
            else:
                shutil.copy(current_file_path, target_file_path)
        # 显示进度
        idx += 1
        if idx % 100 == 0:
            print(f'{idx} / {total} ({idx/total*100}%)')
    print('Dataset compress DONE!')
