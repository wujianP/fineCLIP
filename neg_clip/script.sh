conda activate /discobox/wjpeng/env/clip
cd /discobox/wjpeng/code/202306/fineCLIP/neg_clip/src/
#python -m torch.distributed.launch --nproc_per_node 2 --master_port 5 main.py \

torchrun --nproc_per_node 4 -m training_pix2pix.main \
    --logs="/discobox/wjpeng/ckp/negCLIP/tests" \
    --name="lr5e-5_10ep_warm100_bs256_vitb32" \
    --train-data="/discobox/wjpeng/dataset/pix2pix/train_neg_clip.pth" \
    --val-data="/discobox/wjpeng/dataset/pix2pix/val_neg_clip.pth"  \
    --data-root="/discobox/wjpeng/dataset/pix2pix/clip-filtered-dataset" \
    --batch-size=128 \
    --dataset-type="torch" \
    --epochs=10 \
    --lr=5e-5 \
    --pretrained="openai" \
    --model="ViT-B-32"\
    --workers 14 \
    --warmup 100
