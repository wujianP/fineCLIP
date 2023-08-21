conda activate /discobox/wjpeng/env/clip
cd /discobox/wjpeng/code/202306/fineCLIP/neg_clip/src/training_pix2pix
rm -rf /discobox/wjpeng/ckp/negCLIP/test

torchrun --nproc_per_node 8 --master_port 29500 -m main \
    --logs="/discobox/wjpeng/ckp/negCLIP/test" \
    --name="lr5e-5_10ep_warm100_bs256_vitb32" \
    --train-data="/discobox/wjpeng/dataset/pix2pix/train_neg_clip.pth" \
    --val-data="/discobox/wjpeng/dataset/pix2pix/val_neg_clip.pth"  \
    --data-root="/discobox/wjpeng/dataset/pix2pix/clip-filtered-dataset" \
    --batch-size=32 \
    --dataset-type="torch" \
    --epochs=10 \
    --lr=5e-5 \
    --pretrained="openai" \
    --model="ViT-B-32"\
    --workers 4 \
    --warmup 100
