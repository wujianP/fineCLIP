conda activate /discobox/wjpeng/env/clip
cd /discobox/wjpeng/code/202306/fineCLIP/neg_clip/src/training_pix2pix
rm -rf /DDN_ROOT/wjpeng/ckp/negCLIP/test-tokenize-256

torchrun --nproc_per_node 8 --master_port 29500 -m main \
    --gpu_ids="0,1,2,3,4,5,6,7" \
    --log-freq=10 \
    --logs="/DDN_ROOT/wjpeng/ckp/negCLIP/test-tokenize-256" \
    --name="lr1e-6_5ep_warm100_bs32_vitb32_8gpu" \
    --train-data="/DDN_ROOT/wjpeng/dataset/pix2pix-256/train_neg_clip.pth" \
    --val-data="/DDN_ROOT/wjpeng/dataset/pix2pix-256/val_neg_clip.pth"  \
    --data-root="/DDN_ROOT/wjpeng/dataset/pix2pix-256/clip-filtered-dataset" \
    --batch-size=256 \
    --dataset-type="torch" \
    --epochs=5 \
    --lr=1e-6 \
    --pretrained="openai" \
    --model="ViT-B-32"\
    --workers 8 \
    --warmup 50
