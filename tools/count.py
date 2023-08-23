from PIL import Image
import os
import argparse
import shutil


if __name__ == '__main__':
    dir_path = '/DDN_ROOT/wjpeng/dataset/pix2pix-512/clip-filtered-dataset'
    # 遍历所有子目录
    cnt_dict = {}
    min_num = 10
    max_num = 0
    for dirname in sorted(os.listdir(dir_path)):
        if dirname.endswith('.json'):
            break

        img_cnt = 0
        current_dir_path = os.path.join(dir_path, dirname)
        for filename in sorted(os.listdir(current_dir_path)):
            if filename.endswith('.jpg'):
                img_cnt += 1
        if img_cnt < min_num:
            min_num = img_cnt
        if img_cnt > max_num:
            max_num = img_cnt
        cnt_dict[dirname] = img_cnt

"""
min: 2
max: 8
"""