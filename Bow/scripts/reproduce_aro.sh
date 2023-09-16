cd /discobox/wjpeng/code/202306/fineCLIP/Bow
conda activate /discobox/wjpeng/env/clip

resume=/DDN_ROOT/wjpeng/ckp/negCLIP/mine/negclip_128_1e-6_fastchat/checkpoints/epoch_2.pt
output=/DDN_ROOT/wjpeng/ckp/negCLIP/mine/negclip_128_1e-6_fastchat/evaluate/

#for dataset in VG_Relation VG_Attribution COCO_Order Flickr30k_order
for dataset in VG_Relation VG_Attribution
do
    python3 main_aro.py \
    --dataset $dataset \
    --model-name ViT-B-32 \
    --device cuda \
    --resume $resume \
    --batch-size 128 \
    --num-workers 8 \
    --download \
    --save-scores \
    --output-dir $output
done
