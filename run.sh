while true
do
    python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --master_port=1234 fmow_pretrain.py --output_dir out/joint_sent_less_patch \
    --model mae_vit_large_patch16 --batch_size 16 --accum_iter 8 --norm_pix_loss --output_dir out/joint_sent_less_patch \
    --train_path /atlas/u/yzcong/fmow-sentinel-filtered-csv/train.csv --test_path /atlas/u/yzcong/fmow-sentinel-filtered-csv/val.csv \
    --dataset_type strict_joint --indp_channel --resume /atlas/u/yzcong/mae/out/joint_sent_less_patch/checkpoint-latest.pth \
    --epochs 300 --blr 1.5e-4 --wandb joint_sent_less_patch --num_workers 2
done

# python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --master_port=1234 fmow_pretrain.py --output_dir out/joint_sent_less_patch \
#     --model mae_vit_large_patch16 --batch_size 16 --accum_iter 8 --norm_pix_loss --output_dir out/joint_sent_less_patch \
#     --train_path /atlas/u/yzcong/fmow-sentinel-filtered-csv/train.csv --test_path /atlas/u/yzcong/fmow-sentinel-filtered-csv/val.csv \
#     --dataset_type strict_joint --indp_channel \
#     --epochs 300 --blr 1.5e-4 --num_workers 4