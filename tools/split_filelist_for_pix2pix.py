"""这个文件用于为pix2pix数据集划分适配negCLIP训练框架的标注格式文件"""

import torch

if __name__ == '__main__':
    all_path = '/discobox/wjpeng/dataset/pix2pix/all_neg_clip.pth'
    train_path = '/discobox/wjpeng/dataset/pix2pix/train_neg_clip.pth'
    val_path = '/discobox/wjpeng/dataset/pix2pix/val_neg_clip.pth'

    all_list = torch.load(all_path)
    train_list = all_list[:300000]
    val_list = all_list[300000:]

    print(f'Train #: {len(train_list)}')
    print(f'Val #: {len(val_list)}')

    torch.save(train_list, train_path)
    torch.save(val_list, val_path)
