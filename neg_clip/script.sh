conda activate /discobox/wjpeng/env/clip
cd /discobox/wjpeng/code/202306/fineCLIP/neg_clip/src/training_pix2pix
#rm -rf /discobox/wjpeng/ckp/negCLIP/test

torchrun --nproc_per_node 1 --master_port 29501 -m main \
    --gpu_ids="0" \
    --logs="/discobox/wjpeng/ckp/negCLIP/test" \
    --name="lr1e-6_5ep_warm100_bs256_vitb32_1gpu" \
    --train-data="/discobox/wjpeng/dataset/pix2pix/train_neg_clip.pth" \
    --val-data="/discobox/wjpeng/dataset/pix2pix/val_neg_clip.pth"  \
    --data-root="/discobox/wjpeng/dataset/pix2pix/clip-filtered-dataset" \
    --batch-size=256 \
    --dataset-type="torch" \
    --epochs=5 \
    --lr=1e-6 \
    --pretrained="openai" \
    --model="ViT-B-32"\
    --workers 4 \
    --warmup 100
