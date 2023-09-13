import json
import os

from torch.utils.data import Dataset
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, data_root, ann):
        self.data_root = data_root
        self.ann = ann
        with open(ann, 'r', encoding='utf-8') as file:
            self.data_list = json.load(file)['edited_dirs']
        file.close()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        filename = self.data_list[idx]
        filepath = os.path.join(self.data_root, filename)
        image = Image.open(os.path.join(filepath, 'image.jpg')).convert('RGB')
        ipt_image = Image.open(os.path.join(filepath, 'inpainted_image.jpg')).convert('RGB')
        with open(os.path.join(filepath, 'captions.json'), 'r') as file:
            cap = json.load(file)
            caption = cap['original-image-caption-human']
            ipt_caption = cap['inpainted-image-caption-human']
        file.close()

        return image, ipt_image, caption, ipt_caption
