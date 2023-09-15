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
    --name="negclip_128_1e-6" \
    --lr=1e-6 \
    --pretrained="openai" \
    --model="ViT-B-32"\
    --workers 14 \
    --warmup 50
