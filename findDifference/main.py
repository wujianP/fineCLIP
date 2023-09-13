import argparse


from dataset import MyDataset
# from torch.utils.data import DataLoader


def main():
    dataset = MyDataset(
        data_root=args.data_root,
        ann=args.ann
    )

    for i in range(len(dataset)):
        image, ipt_image, caption, ipt_caption = dataset[i]
        from IPython import embed
        embed()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data/remove-stable-diffusion-filtered')
    parser.add_argument('--ann', type=str, default='./data/remove-stable-diffusion-refine_caption_outputs.json')
    args = parser.parse_args()
    main()
