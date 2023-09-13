import argparse
import torch
import open_clip
import wandb
wandb.login()

from dataset import MyDataset
from PIL import Image
# from torch.utils.data import DataLoader


def main():

    dataset = MyDataset(
        data_root=args.data_root,
        ann=args.ann
    )

    # model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    # tokenizer = open_clip.get_tokenizer('ViT-B-32')
    # model = model.cuda()

    for i in range(len(dataset)):
        image, ipt_image, caption, ipt_caption = dataset[i]
        # image = preprocess(image.resize((512, 512)))
        # ipt_image = preprocess(ipt_image.resize((512, 512)))
        #
        # images = torch.stack([image, ipt_image], dim=0).cuda()
        # text = tokenizer([caption]).cuda()

        # with torch.no_grad():
        #     image_features = model.encode_image(images)
        #     text_features = model.encode_text(text)
        #
        #     image_features /= image_features.norm(dim=-1, keepdim=True)
        #     text_features /= text_features.norm(dim=-1, keepdim=True)
        #
        #     img_sim = (image_features @ image_features.T) * 100
        #     # print(img_sim)
        #
        #     image_probs = (100.0 * text_features @ image_features.T).softmax(dim=-1)

        from IPython import embed
        embed()

        run.log({'Find Difference': [wandb.Image(image, caption=caption), wandb.Image(ipt_image, caption=ipt_image)]})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data/remove-stable-diffusion-filtered')
    parser.add_argument('--ann', type=str, default='./data/remove-stable-diffusion-refine_caption_outputs.json')
    args = parser.parse_args()
    run = wandb.init('find difference')
    main()
