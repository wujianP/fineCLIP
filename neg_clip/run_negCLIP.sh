# >>> OFFICIAL >>>
conda activate /discobox/wjpeng/env/clip
cd /discobox/wjpeng/code/202306/fineCLIP/neg_clip/src/training
git pull

CUDA_VISIBLE_DEVICES=0 python -m main \
    --data-root /DDN_ROOT/wjpeng/dataset \
    --train-data="/discobox/wjpeng/code/202306/fineCLIP/neg_clip/data/train_neg_clip.tsv" \
    --val-data="/discobox/wjpeng/code/202306/fineCLIP/neg_clip/data/valid_neg_clip.tsv"  \
    --logs="/DDN_ROOT/wjpeng/ckp/negCLIP/official" \
    --batch-size=128 \
    --epochs=5 \
    --name="negclip_128_5e-7" \
    --lr=5e-7 \
    --pretrained="openai" \
    --model="ViT-B-32"\
    --workers 14 \
    --warmup 50


# >>> FASTCHAT >>>
conda activate /discobox/wjpeng/env/clip
cd /discobox/wjpeng/code/202306/fineCLIP/neg_clip/src/training
git pull

CUDA_VISIBLE_DEVICES=1 python -m main \
    --data-root /DDN_ROOT/wjpeng/dataset \
    --train-data="/discobox/wjpeng/code/202306/fineCLIP/neg_clip/data/train_neg_clip_fastchat.pth" \
    --val-data="/discobox/wjpeng/code/202306/fineCLIP/neg_clip/data/valid_neg_clip.tsv"  \
    --logs="/DDN_ROOT/wjpeng/ckp/negCLIP/mine" \
    --batch-size=128 \
    --epochs=5 \
    --name="negclip_128_5e-7_fastchat" \
    --lr=5e-7 \
    --pretrained="openai" \
    --model="ViT-B-32"\
    --workers 14 \
    --warmup 50
