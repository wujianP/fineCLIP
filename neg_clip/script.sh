conda activate /discobox/wjpeng/env/openVCLIP
cd /discobox/wjpeng/code/202306/fineCLIP/neg_clip/src/training_pix2pix

CUDA_VISIBLE_DEVICES=3
python main.py \
    --logs="/discobox/wjpeng/ckp/negCLIP/ablate/lr" \
    --name="lr5e-5_10ep_warm100_bs256_vitb32" \
    --train-data="/discobox/wjpeng/dataset/pix2pix/train_neg_clip.pth" \
    --val-data="/discobox/wjpeng/dataset/pix2pix/val_neg_clip.pth"  \
    --data-root="/discobox/wjpeng/dataset/pix2pix/clip-filtered-dataset" \
    --batch-size=256 \
    --dataset-type="torch" \
    --epochs=10 \
    --lr=5e-5 \
    --pretrained="openai" \
    --model="ViT-B-32"\
    --workers 14 \
    --warmup 100
