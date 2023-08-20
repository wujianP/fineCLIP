"""这个文件用于为pix2pix数据集构造适配negCLIP训练框架的.tsv格式文件"""
import glob
import json
import os

import torch

if __name__ == '__main__':
    out_path = '/discobox/wjpeng/dataset/pix2pix/all_neg_clip.pth'
    pix_root = '/discobox/wjpeng/dataset/pix2pix/clip-filtered-dataset'

    data_list = []
    for filename in sorted(os.listdir(pix_root)):
        if filename.startswith('0'):
            file_path = os.path.join(pix_root, filename)
            # read text and hard text
            with open(os.path.join(file_path, 'prompt.json'), 'r') as f:
                prompt = json.load(f)
                text = prompt['input']
                hard_text = prompt['output']
            # read image and hard images
            images, hard_images = [], []
            for img in os.listdir(file_path):
                if img.endswith('0.jpg'):
                    images.append(img)
                if img.endswith('1.jpg'):
                    hard_images.append(img)
            # construct data
            data = {
                'filename': filename,
                'text': text,
                'hard_text': hard_text,
                'images': images,
                'hard_images': hard_images
            }
            data_list.append(data)
            from IPython import embed
            embed()

    torch.save(data_list, out_path)
