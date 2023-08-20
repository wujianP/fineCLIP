conda activate /discobox/wjpeng/env/openVCLIP
cd /discobox/wjpeng/code/202306/fineCLIP/neg_clip/src/training_pix2pix

CUDA_VISIBLE_DEVICES=0 python main.py \
    --train-data="/discobox/wjpeng/dataset/pix2pix/train_neg_clip.pth" \
    --batch-size=256 \
    --epochs=5 \
    --name="negclip_test" \
    --lr=1e-6 \
    --val-data="./discobox/wjpeng/dataset/pix2pix/val_neg_clip.pth"  \
    --logs="/discobox/wjpeng/ckp/negCLIP/test" \
    --pretrained="openai" \
    --model="ViT-B-32"\
    --workers 14 \
    --warmup 50