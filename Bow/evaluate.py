import pandas as pd
import argparse

from torch.utils.data import DataLoader
from model_zoo.clip_models import CLIPWrapper
from open_clip import create_model_and_transforms
from dataset_zoo import VG_Relation, VG_Attribution


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, default='ViT-B-32')
    parser.add_argument('--resume', type=str)
    parser.add_argument('--data-root', type=str, default='/DDN_ROOT/wjpeng/dataset/ARO',
                        help="VG-Relation and VG-Attribution images here")
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--num-workers', type=int, default=8)
    args = parser.parse_args()

    # Get model
    model, _, preprocess = create_model_and_transforms(
        model_name=args.model_name,
        pretrained=args.resume,
        device="cuda",
    )
    model = CLIPWrapper(model, "cuda")

    # Get the VG-R dataset
    vgr_dataset = VG_Relation(image_preprocess=preprocess, download=True, root_dir=args.data_root)
    vgr_loader = DataLoader(vgr_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Compute the scores for each test case
    vgr_scores = model.get_retrieval_scores_batched(vgr_loader)

    # Evaluate the macro accuracy
    vgr_records = vgr_dataset.evaluate_scores(vgr_scores)
    symmetric = ['adjusting', 'attached to', 'between', 'bigger than', 'biting', 'boarding', 'brushing', 'chewing', 'cleaning', 'climbing', 'close to', 'coming from', 'coming out of', 'contain', 'crossing', 'dragging', 'draped over', 'drinking', 'drinking from', 'driving', 'driving down', 'driving on', 'eating from', 'eating in', 'enclosing', 'exiting', 'facing', 'filled with', 'floating in', 'floating on', 'flying', 'flying above', 'flying in', 'flying over', 'flying through', 'full of', 'going down', 'going into', 'going through', 'grazing in', 'growing in', 'growing on', 'guiding', 'hanging from', 'hanging in', 'hanging off', 'hanging over', 'higher than', 'holding onto', 'hugging', 'in between', 'jumping off', 'jumping on', 'jumping over', 'kept in', 'larger than', 'leading', 'leaning over', 'leaving', 'licking', 'longer than', 'looking in', 'looking into', 'looking out', 'looking over', 'looking through', 'lying next to', 'lying on top of', 'making', 'mixed with', 'mounted on', 'moving', 'on the back of', 'on the edge of', 'on the front of', 'on the other side of', 'opening', 'painted on', 'parked at', 'parked beside', 'parked by', 'parked in', 'parked in front of', 'parked near', 'parked next to', 'perched on', 'petting', 'piled on', 'playing', 'playing in', 'playing on', 'playing with', 'pouring', 'reaching for', 'reading', 'reflected on', 'riding on', 'running in', 'running on', 'running through', 'seen through', 'sitting behind', 'sitting beside', 'sitting by', 'sitting in front of', 'sitting near', 'sitting next to', 'sitting under', 'skiing down', 'skiing on', 'sleeping in', 'sleeping on', 'smiling at', 'sniffing', 'splashing', 'sprinkled on', 'stacked on', 'standing against', 'standing around', 'standing behind', 'standing beside', 'standing in front of', 'standing near', 'standing next to', 'staring at', 'stuck in', 'surrounding', 'swimming in', 'swinging', 'talking to', 'topped with', 'touching', 'traveling down', 'traveling on', 'tying', 'typing on', 'underneath', 'wading in', 'waiting for', 'walking across', 'walking by', 'walking down', 'walking next to', 'walking through', 'working in', 'working on', 'worn on', 'wrapped around', 'wrapped in', 'by', 'of', 'near', 'next to', 'with', 'beside', 'on the side of', 'around']
    df = pd.DataFrame(vgr_records)
    df = df[~df.Relation.isin(symmetric)]
    print(f"VG-Relation Macro Accuracy: {df.Accuracy.mean()}")

    # Get the VG-A dataset
    vga_dataset = VG_Attribution(image_preprocess=preprocess, download=True, root_dir=args.data_root)
    vga_loader = DataLoader(vga_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    # Compute the scores for each test case
    vga_scores = model.get_retrieval_scores_batched(vga_loader)
    # Evaluate the macro accuracy
    vga_records = vga_dataset.evaluate_scores(vga_scores)
    df = pd.DataFrame(vga_records)
    print(f"VG-Attribution Macro Accuracy: {df.Accuracy.mean()}")
