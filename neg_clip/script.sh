conda activate /discobox/wjpeng/env/clip
cd /discobox/wjpeng/code/202306/fineCLIP/neg_clip/src/training_pix2pix
rm -rf /DDN_ROOT/wjpeng/ckp/negCLIP/test

torchrun --nproc_per_node 2 --master_port 29500 -m main \
    --gpu_ids="0,1" \
    --log-freq=10 \
    --logs="/DDN_ROOT/wjpeng/ckp/negCLIP/test" \
    --name="lr1e-6_5ep_warm100_bs256_vitb32_2gpu_" \
    --train-data="/DDN_ROOT/wjpeng/dataset/pix2pix-512/train_neg_clip.pth" \
    --val-data="/DDN_ROOT/wjpeng/dataset/pix2pix-512/val_neg_clip.pth"  \
    --data-root="/DDN_ROOT/wjpeng/dataset/pix2pix-512/clip-filtered-dataset" \
    --batch-size=128 \
    --dataset-type="torch" \
    --epochs=5 \
    --lr=1e-6 \
    --pretrained="openai" \
    --model="ViT-B-32"\
    --workers 10 \
    --warmup 100
