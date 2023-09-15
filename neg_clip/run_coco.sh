conda activate /discobox/wjpeng/env/clip
cd /discobox/wjpeng/code/202306/fineCLIP/neg_clip/src/training
git pull

CUDA_VISIBLE_DEVICES=0 python -m training.main \
    --train-data="data/train_neg_clip.tsv" \
    --val-data="data/valid_neg_clip.tsv"  \
    --batch-size=256 \
    --epochs=5 \
    --name="negclip_256_1e-6" \
    --lr=1e-6 \
    --logs="./logs/negCLIP/" \
    --pretrained="openai" \
    --model="ViT-B-32"\
    --workers 14 \
    --warmup 50
