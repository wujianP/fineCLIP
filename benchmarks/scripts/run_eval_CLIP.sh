conda activate /discobox/wjpeng/env/clip
cd /discobox/wjpeng/code/202306/fineCLIP/benchmarks/EqBen/example
git pull
export CUDA_VISIBLE_DEVICES=1
python eqben_eval_CLIP.py --eval_data winoground --batch_size 256 \
--resume /DDN_ROOT/wjpeng/ckp/negCLIP/mine/negclip_128_1e-6_fastchat/checkpoints/epoch_5.pt \
--load_file_basename fastchat_vitb_32
